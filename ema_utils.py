import copy
import torch
import torch.nn.functional as F


def create_ema_teacher(student):
    """Create a frozen EMA teacher initialized from the student."""
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, student, decay=0.999, global_step=None):
    """Update teacher parameters by EMA and copy student buffers.

    BatchNorm buffers are copied directly from the student so the teacher uses
    current running statistics instead of lagging far behind them.
    """
    if global_step is None:
        ema_decay = float(decay)
    else:
        ema_decay = min(float(decay), 1.0 - 1.0 / float(global_step + 1))

    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher_params.items():
        teacher_param.mul_(ema_decay).add_(
            student_params[name].detach(), alpha=1.0 - ema_decay
        )

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher_buffers.items():
        teacher_buffer.copy_(student_buffers[name].detach())

    teacher.eval()


@torch.no_grad()
def select_dual_view_pseudo_labels(logits1, logits2, thresholds):
    """Keep a pseudo label only when both weak views agree and pass threshold.

    The original CAST class-adaptive threshold is preserved. No temperature
    scaling, class-distribution correction, prototype weighting or extra sample
    weighting is applied in this ablation.
    """
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1)

    _, pred1 = probs1.max(dim=1)
    _, pred2 = probs2.max(dim=1)

    mean_probs = (probs1 + probs2) * 0.5
    confidence, pseudo_targets = mean_probs.max(dim=1)

    agreement = pred1.eq(pred2) & pred1.eq(pseudo_targets)
    thresholds = thresholds.to(device=logits1.device, dtype=confidence.dtype)
    sample_threshold = thresholds.index_select(0, pseudo_targets)
    reliable = agreement & (confidence >= sample_threshold)

    return pseudo_targets, reliable.float(), int(agreement.sum().item())
