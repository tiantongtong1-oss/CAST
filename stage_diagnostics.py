"""Read-only stage counts and validation-only quality; labels never enter gating."""
import csv
import os
import numpy as np

NAMES = ['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']


def stage_masks(agreement, confidence, distance, vote, probability, prototype,
                final, supervised, anchors):
    """Rescue funnel is cumulative, while direct confidence is a separate branch."""
    candidate = agreement & ~confidence
    near = candidate & distance
    voted = near & vote
    prob = voted & probability
    proto = prob & prototype
    return dict(teacher_raw=np.ones_like(agreement,dtype=bool), agreement=agreement,
                high_confidence=confidence, rescue_candidate=candidate,
                rescue_distance=near, rescue_vote=voted, rescue_probability=prob,
                rescue_prototype=proto, final_accept=final,
                supervised=supervised, anchors=anchors)


class StageStats:
    def __init__(self, classes=7, quality=False):
        self.classes, self.quality = classes, quality
        self.counts, self.correct, self.matrices = {}, {}, {}
        self.support = np.zeros(classes,dtype=np.int64)

    def add(self, predictions, masks, truth=None):
        pred=np.asarray(predictions,dtype=np.int64)
        if self.quality:
            if truth is None: raise ValueError('validation truth required for quality report')
            truth=np.asarray(truth,dtype=np.int64)
            self.support += np.bincount(truth,minlength=self.classes)
        elif truth is not None:
            raise ValueError('training counters must not consume target labels')
        for stage,mask in masks.items():
            mask=np.asarray(mask,dtype=bool)
            self.counts.setdefault(stage,np.zeros(self.classes,dtype=np.int64))
            self.counts[stage] += np.bincount(pred[mask],minlength=self.classes)
            if self.quality:
                cm=np.bincount(truth[mask]*self.classes+pred[mask],minlength=self.classes**2).reshape(self.classes,self.classes)
                self.matrices.setdefault(stage,np.zeros_like(cm))
                self.matrices[stage] += cm

    def report(self, path, epoch, split):
        fresh=not os.path.exists(path)
        with open(path,'a',newline='') as handle:
            writer=csv.writer(handle)
            if fresh: writer.writerow(['epoch','split','stage','class','selected','correct','true_support','precision','correct_coverage'])
            for stage,count in self.counts.items():
                print('[StageDiag %s Epoch %d] %s Counts: %s' % (split,epoch,stage,count.tolist()))
                correct=self.matrices[stage].diagonal() if self.quality else None
                if self.quality:
                    precision=np.divide(correct,count,out=np.full(self.classes,np.nan),where=count>0)
                    coverage=np.divide(correct,self.support,out=np.full(self.classes,np.nan),where=self.support>0)
                    print('  Precision: %s Correct_Coverage: %s' %
                          (np.array2string(precision,precision=4),np.array2string(coverage,precision=4)))
                for c in range(self.classes):
                    writer.writerow([epoch,split,stage,NAMES[c],int(count[c]),
                                     int(correct[c]) if self.quality else '',
                                     int(self.support[c]) if self.quality else '',
                                     precision[c] if self.quality else '',coverage[c] if self.quality else ''])
        if self.quality:
            matrix_path=path.replace('.csv','_confusion.csv')
            fresh=not os.path.exists(matrix_path)
            with open(matrix_path,'a',newline='') as handle:
                writer=csv.writer(handle)
                if fresh: writer.writerow(['epoch','stage','true_class']+NAMES)
                for stage,cm in self.matrices.items():
                    for c,row in enumerate(cm): writer.writerow([epoch,stage,NAMES[c]]+row.tolist())
            print('[StageDiag validation] Teacher raw confusion (rows=true, columns=predicted):\n%s' % self.matrices['teacher_raw'])


def validation_diagnostics(teacher, hook, loader, radius_bank, source_bank, config,
                           thresholds, roles, epoch, args):
    import torch
    import torch.nn.functional as F
    from ema_utils import select_dual_view_pseudo_labels
    from adaptive_distance_vote import audit_adaptive_vote
    from competition_gate import competition_audit, supervision_mask
    device=next(teacher.parameters()).device
    stats=StageStats(7,quality=True)
    teacher.eval()
    with torch.no_grad():
        for weak1,weak2,truth,ids in loader:
            logits1,_=teacher(weak1.to(device),None,None,'test','target')
            f1=hook.output.detach().clone()
            logits2,_=teacher(weak2.to(device),None,None,'test','target')
            f2=hook.output.detach()
            features=F.normalize(F.normalize(f1,dim=1)+F.normalize(f2,dim=1),dim=1)
            labels,conf,_=select_dual_view_pseudo_labels(logits1,logits2,thresholds)
            confidence=conf.bool(); agreement=labels.eq(logits2.argmax(1))
            region=radius_bank.in_distribution(features,labels)
            distance=torch.zeros_like(agreement); vote=distance.clone()
            probability=distance.clone(); prototype=distance.clone(); rescue=distance.clone()
            idx=torch.where(agreement & ~confidence)[0]
            if epoch>=args.radius_rescue_warmup_epochs and len(idx):
                y=labels[idx].cpu().numpy(); x=features[idx].cpu().numpy()
                evidence=source_bank.evaluate(x,ids[idx.cpu()].numpy(),config,y)
                audit=audit_adaptive_vote(evidence,y)
                comp=competition_audit(F.softmax(logits1[idx],dim=1).cpu().numpy(),
                    F.softmax(logits2[idx],dim=1).cpu().numpy(),x,y,
                    radius_bank.class_means.cpu().numpy(),radius_bank.distribution_initialized.cpu().numpy(),
                    evidence['votes'],evidence['available'],args.rescue_probability_margin,
                    args.rescue_prototype_margin,args.rescue_vote_margin)
                distance[idx]=torch.as_tensor(evidence['available'],device=device)
                vote[idx]=torch.as_tensor(audit['pass_mask'],device=device)
                probability[idx]=torch.as_tensor(comp['probability_gap']>=args.rescue_probability_margin,device=device)
                prototype[idx]=torch.as_tensor(comp['prototype_gap']>args.rescue_prototype_margin,device=device)
                rescue[idx]=vote[idx] & torch.as_tensor(comp['pass_mask'],device=device)
            final=confidence | rescue
            minconf=torch.minimum(F.softmax(logits1,dim=1).max(1).values,F.softmax(logits2,dim=1).max(1).values)
            # Separate validation IDs/history; never touch the training role tracker.
            anchors=roles.observe(ids,labels,final & region,minconf,epoch)
            learned=supervision_mask(confidence,rescue,anchors)
            values=[agreement,confidence,distance,vote,probability,prototype,final,learned,anchors]
            masks=stage_masks(*[v.cpu().numpy() for v in values])
            stats.add(labels.cpu().numpy(),masks,truth.numpy())
    return stats
