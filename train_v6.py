from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cast_v6.ccdr import class_volume_weights, classifier_modulation_loss
from cast_v6.data import build_fer2013, build_rafdb, print_dataset_summary
from cast_v6.ddrl import ddrl_loss
from cast_v6.metrics import JsonlLogger, evaluate, prediction_health, print_pseudo_log
from cast_v6.model import (
    EMATeacher,
    FERNet,
    freeze_bn_stats,
    load_source_checkpoint,
    recalibrate_backbone_bn,
)
from cast_v6.pseudo import build_pseudo_bank


def parse_args():
    p = argparse.ArgumentParser("CAST v6 - modular dual-view EMA + DDRL + CCDR")
    p.add_argument("--source-root", required=True)
    p.add_argument("--target-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "mobilenet_v2"])
    p.add_argument("--fer-folder-order", default="cast", choices=["cast", "kaggle"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=1314)

    p.add_argument("--target-lr", type=float, default=2e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.25)
    p.add_argument("--lr-gamma", type=float, default=0.97)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--w1", type=float, default=4.0)
    p.add_argument("--w2", type=float, default=0.03,
                   help="Stable target DDRL weight. Paper reports beta=0.3; raise only after the pipeline is stable.")
    p.add_argument("--w3", type=float, default=0.1)
    p.add_argument("--target-lambda", type=float, default=0.35)
    p.add_argument("--ddrl-min-class-samples", type=int, default=2)
    p.add_argument("--ddrl-min-classes", type=int, default=3)
    p.add_argument("--affinity-warmup", type=int, default=5)
    p.add_argument("--affinity-mid-epochs", type=int, default=5)
    p.add_argument("--affinity-mid-weight", type=float, default=0.01)

    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="<=0 fits scalar temperature on RAF-DB test; >0 uses fixed value")
    p.add_argument("--phi", type=float, default=1.4)
    p.add_argument("--threshold-cap", type=float, default=0.9)
    p.add_argument("--pseudo-min-margin", type=float, default=0.0)
    p.add_argument("--pseudo-max-entropy", type=float, default=1.0)
    p.add_argument("--pseudo-confidence-floor", type=float, default=0.0)

    p.add_argument("--bn-recalibrate-batches", type=int, default=64)
    p.add_argument("--bn-recalibrate-momentum", type=float, default=0.03)
    p.add_argument("--debug-target-labels", action="store_true",
                   help="Log pseudo accuracy / per-epoch target metrics. Never used in optimization.")
    p.add_argument("--abort-on-collapse", action="store_true")
    p.add_argument("--model-dir", default="")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(ds, batch_size, workers, shuffle, drop_last=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )


@torch.no_grad()
def collect_logits_labels(model, loader, device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        out, _ = model(x)
        logits.append(out.detach())
        labels.append(y.detach())
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(model, loader, device):
    logits, labels = collect_logits_labels(model, loader, device)
    before = float(F.cross_entropy(logits, labels).item())
    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp().clamp(0.25, 4.0)
        loss = F.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = float(log_t.detach().exp().clamp(0.25, 4.0).item())
    after = float(F.cross_entropy(logits / t, labels).item())
    return t, before, after


def target_ce(logits, pseudo, weights, selected):
    if int(selected.sum().item()) == 0:
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits[selected], pseudo[selected], reduction="none")
    w = weights[selected].clamp_min(1e-6)
    return (ce * w).sum() / w.sum()


def ddrl_weight_for_epoch(args, epoch, active_classes, health):
    if active_classes < args.ddrl_min_classes:
        return 0.0
    if health["selected_ratio"] < 0.02 or health["selected_classes"] < args.ddrl_min_classes:
        return 0.0
    if epoch < args.affinity_warmup:
        return 0.0
    if epoch < args.affinity_warmup + args.affinity_mid_epochs:
        return min(args.w2, args.affinity_mid_weight)
    return args.w2


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.model_dir:
        args.model_dir = os.path.join("models", "cast_resnet50_v6", stamp)
    os.makedirs(args.model_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(args.model_dir, "metrics.jsonl"))

    print("CAST v6 configuration:", json.dumps(vars(args), sort_keys=True))
    print("Device:", device)

    source_train = build_rafdb(args.source_root, "train", mode="train")
    source_eval = build_rafdb(args.source_root, "test", mode="eval")
    target_train = build_fer2013(args.target_root, "train", mode="train", folder_order=args.fer_folder_order)
    target_pseudo = build_fer2013(args.target_root, "train", mode="pseudo", folder_order=args.fer_folder_order)
    target_test = build_fer2013(args.target_root, "test", mode="eval", folder_order=args.fer_folder_order)
    try:
        target_val = build_fer2013(args.target_root, "val", mode="eval", folder_order=args.fer_folder_order)
    except FileNotFoundError:
        target_val = None

    print_dataset_summary("RAF train", source_train)
    print_dataset_summary("RAF test", source_eval)
    print_dataset_summary("FER train", target_train)
    if target_val is not None:
        print_dataset_summary("FER val", target_val)
    else:
        print("[Data][WARN] FER validation split not found. Test metrics will NOT be used for checkpoint selection.")
    print_dataset_summary("FER test", target_test)

    source_train_loader = make_loader(source_train, args.batch_size, args.workers, True, drop_last=True)
    source_eval_loader = make_loader(source_eval, args.eval_batch_size, args.workers, False)
    target_train_loader = make_loader(target_train, args.batch_size, args.workers, True, drop_last=True)
    target_pseudo_loader = make_loader(target_pseudo, args.eval_batch_size, args.workers, False)
    target_test_loader = make_loader(target_test, args.eval_batch_size, args.workers, False)
    target_val_loader = None if target_val is None else make_loader(target_val, args.eval_batch_size, args.workers, False)

    student = FERNet(args.backbone, num_classes=7, pretrained=False, use_logit_bn=True).to(device)
    load_info = load_source_checkpoint(student, args.checkpoint, device)
    print("Loaded source checkpoint: %s coverage=%.1f%% missing=%d unexpected=%d" % (
        args.checkpoint, load_info["coverage"] * 100.0,
        len(load_info["missing"]), len(load_info["unexpected"])
    ))

    if args.temperature <= 0:
        temperature, nll_before, nll_after = fit_temperature(student, source_eval_loader, device)
        print("Source-validation temperature %.4f NLL %.4f -> %.4f" % (temperature, nll_before, nll_after))
    else:
        temperature = args.temperature
        print("Using fixed teacher temperature %.4f" % temperature)

    start = evaluate(student, target_test_loader, device)
    print("[Source -> Target] accuracy %.4f class_acc=%s predicted=%s" % (
        start["acc"], [round(x, 4) for x in start["class_acc"]], start["predicted"]
    ))

    seen = recalibrate_backbone_bn(
        student, target_pseudo_loader, device,
        batches=args.bn_recalibrate_batches,
        momentum=args.bn_recalibrate_momentum,
    )
    after_bn = evaluate(student, target_test_loader, device)
    print("Target backbone-BN recalibration batches=%d" % seen)
    print("[Target start after BN] accuracy %.4f class_acc=%s predicted=%s" % (
        after_bn["acc"], [round(x, 4) for x in after_bn["class_acc"]], after_bn["predicted"]
    ))

    teacher = EMATeacher.from_student(student, decay=args.ema_decay)
    freeze_bn_stats(student, freeze_affine=False)
    freeze_bn_stats(teacher.model, freeze_affine=False)

    optimizer = torch.optim.Adam([
        {"params": student.feature.parameters(), "lr": args.target_lr * args.backbone_lr_mult},
        {"params": list(student.fc.parameters()) + list(student.bn.parameters()), "lr": args.target_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    best_val = -1.0
    collapse_streak = 0

    for epoch in range(args.epochs):
        bank = build_pseudo_bank(
            teacher.model,
            target_pseudo_loader,
            dataset_size=len(target_pseudo),
            num_classes=7,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            phi=args.phi,
            threshold_cap=args.threshold_cap,
            temperature=temperature,
            min_margin=args.pseudo_min_margin,
            max_entropy=args.pseudo_max_entropy,
            confidence_floor=args.pseudo_confidence_floor,
            debug_target_labels=args.debug_target_labels,
        )
        print_pseudo_log(epoch, bank)
        health = prediction_health(bank.predicted_counts, bank.selected_counts, len(target_pseudo))
        print("[Epoch %d][Health] max_pred_ratio=%.4f pred_entropy=%.4f selected_classes=%d selected_ratio=%.4f status=%s" % (
            epoch, health["max_pred_ratio"], health["pred_entropy"], health["selected_classes"],
            health["selected_ratio"], health["status"]
        ))

        if health["status"] != "OK":
            collapse_streak += 1
        else:
            collapse_streak = 0
        if args.abort_on_collapse and collapse_streak >= 2:
            print("[ABORT] Health checks failed for two consecutive epochs. Stop before self-training amplifies collapse.")
            break

        student.train()
        freeze_bn_stats(student, freeze_affine=False)
        source_iter = iter(source_train_loader)

        sums = {"src_ce": 0.0, "tgt_ce": 0.0, "cls": 0.0, "ddrl": 0.0,
                "ddrl_intra": 0.0, "ddrl_inter": 0.0, "cscm": 0.0,
                "cscm_cos": 0.0, "total": 0.0, "grad": 0.0, "w2": 0.0}
        batches = 0
        last_ddrl = {"intra": 0.0, "inter": 0.0, "loss": 0.0,
                     "intra_classes": [], "inter_classes": [], "active_classes": 0}
        eta_acc = torch.zeros(7)
        eta_batches = 0
        last_volume = None
        ema_decay_used = 0.0

        for target_batch in target_train_loader:
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_train_loader)
                source_batch = next(source_iter)
            sw1, _, _, sy, _ = source_batch
            _, _, tx, _, tidx = target_batch
            sw1 = sw1.to(device, non_blocking=True)
            sy = sy.to(device, non_blocking=True)
            tx = tx.to(device, non_blocking=True)
            tidx = tidx.long()

            py = bank.labels[tidx].to(device)
            pw = bank.weights[tidx].to(device)
            pm = bank.selected[tidx].to(device)

            slogits, sfeat = student(sw1)
            tlogits, tfeat = student(tx)

            src_ce = F.cross_entropy(slogits, sy)
            tgt_ce_value = target_ce(tlogits, py, pw, pm)
            cls_loss = src_ce + args.target_lambda * tgt_ce_value
            cscm_loss, mean_cos = classifier_modulation_loss(student.fc.weight)

            if int(pm.sum().item()) > 0:
                tfeat_sel = tfeat[pm]
                py_sel = py[pm]
                combined_feat = torch.cat([sfeat.detach(), tfeat_sel.detach()], dim=0)
                combined_y = torch.cat([sy, py_sel], dim=0)
                eta, volume_info = class_volume_weights(combined_feat, combined_y, num_classes=7)
                ddrl_value, ddrl_info = ddrl_loss(
                    sfeat, sy, tfeat_sel, py_sel, eta,
                    num_classes=7,
                    min_class_samples=args.ddrl_min_class_samples,
                )
                last_volume = volume_info
                eta_acc += eta.detach().cpu()
                eta_batches += 1
            else:
                ddrl_value = sfeat.sum() * 0.0
                ddrl_info = {"intra": 0.0, "inter": 0.0, "loss": 0.0,
                             "intra_classes": [], "inter_classes": [], "active_classes": 0}

            w2_eff = ddrl_weight_for_epoch(args, epoch, ddrl_info["active_classes"], health)
            loss = args.w1 * cls_loss + w2_eff * ddrl_value + args.w3 * cscm_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip).item())
            optimizer.step()
            ema_decay_used = teacher.update(student)

            sums["src_ce"] += float(src_ce.detach().item())
            sums["tgt_ce"] += float(tgt_ce_value.detach().item())
            sums["cls"] += float(cls_loss.detach().item())
            sums["ddrl"] += float(ddrl_value.detach().item())
            sums["ddrl_intra"] += float(ddrl_info["intra"])
            sums["ddrl_inter"] += float(ddrl_info["inter"])
            sums["cscm"] += float(cscm_loss.detach().item())
            sums["cscm_cos"] += float(mean_cos)
            sums["total"] += float(loss.detach().item())
            sums["grad"] += grad_norm
            sums["w2"] += w2_eff
            batches += 1
            last_ddrl = ddrl_info

        scheduler.step()
        denom = max(1, batches)
        avg = {k: v / denom for k, v in sums.items()}
        avg_eta = (eta_acc / max(1, eta_batches)).tolist()
        volume_text = "None" if last_volume is None else [
            None if v is None else round(float(v), 3) for v in last_volume["volume"]
        ]
        print("[Epoch %d][CCDR] eta=%s volume(last_batch)=%s classifier_cos=%.4f" % (
            epoch, [round(x, 4) for x in avg_eta], volume_text, avg["cscm_cos"]
        ))
        print("[Epoch %d][DDRL] intra=%.5f inter=%.5f loss=%.5f active(last_batch)=%s w2=%.4f" % (
            epoch, avg["ddrl_intra"], avg["ddrl_inter"], avg["ddrl"],
            last_ddrl["intra_classes"], avg["w2"]
        ))
        print("[Epoch %d][Train] source_ce=%.4f target_ce=%.4f cls=%.4f ddrl=%.4f cscm=%.4f total=%.4f grad_norm=%.4f lr_backbone=%.7f lr_head=%.7f ema_decay=%.6f" % (
            epoch, avg["src_ce"], avg["tgt_ce"], avg["cls"], avg["ddrl"], avg["cscm"],
            avg["total"], avg["grad"], optimizer.param_groups[0]["lr"],
            optimizer.param_groups[1]["lr"], ema_decay_used
        ))

        student_test = evaluate(student, target_test_loader, device)
        teacher_test = evaluate(teacher.model, target_test_loader, device)
        print("[Epoch %d][Eval][Student] acc=%.4f class_acc=%s predicted=%s" % (
            epoch, student_test["acc"], [round(x, 4) for x in student_test["class_acc"]], student_test["predicted"]
        ))
        print("[Epoch %d][Eval][EMA] acc=%.4f class_acc=%s predicted=%s" % (
            epoch, teacher_test["acc"], [round(x, 4) for x in teacher_test["class_acc"]], teacher_test["predicted"]
        ))

        record = {
            "epoch": epoch,
            "pseudo": {
                "thresholds": bank.thresholds.tolist(),
                "predicted": bank.predicted_counts,
                "selected": bank.selected_counts,
                "agreement": bank.agreement_rate,
                "selected_ratio": bank.selected_ratio,
                "pseudo_acc": bank.pseudo_accuracy,
                "pseudo_class_acc": bank.pseudo_class_accuracy,
            },
            "health": health,
            "train": avg,
            "ddrl_last": last_ddrl,
            "ccdr_eta": avg_eta,
            "student_test": student_test,
            "ema_test": teacher_test,
        }
        logger.write(record)

        save = {
            "student": student.state_dict(),
            "teacher": teacher.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "temperature": temperature,
            "config": vars(args),
        }
        torch.save(save, os.path.join(args.model_dir, "last.pth"))

        if target_val_loader is not None:
            val_metrics = evaluate(teacher.model, target_val_loader, device)
            print("[Epoch %d][Val][EMA] acc=%.4f class_acc=%s" % (
                epoch, val_metrics["acc"], [round(x, 4) for x in val_metrics["class_acc"]
            ))
            if val_metrics["acc"] > best_val:
                best_val = val_metrics["acc"]
                torch.save(save, os.path.join(args.model_dir, "best_val.pth"))
                print("[Epoch %d] best_val=%.4f" % (epoch, best_val))

    print("Training finished. Logs:", os.path.join(args.model_dir, "metrics.jsonl"))
    print("Checkpoint:", os.path.join(args.model_dir, "last.pth"))


if __name__ == "__main__":
    main()
