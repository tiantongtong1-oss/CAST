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
    """Update teacher parameters and floating buffers with a fixed EMA decay.

    The teacher is initialized from the student, so an early-step decay warmup is
    unnecessary and makes the teacher follow noisy target updates too quickly.
    Floating-point buffers (notably BatchNorm running_mean/running_var) receive
    the same EMA update. Integer counters such as num_batches_tracked are copied.
    ``global_step`` is retained only for call-site compatibility.
    """
    del global_step
    ema_decay = float(decay)

    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher_params.items():
        teacher_param.mul_(ema_decay).add_(
            student_params[name].detach(), alpha=1.0 - ema_decay
        )

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher_buffers.items():
        student_buffer = student_buffers[name].detach()
        if torch.is_floating_point(teacher_buffer):
            teacher_buffer.mul_(ema_decay).add_(
                student_buffer.to(dtype=teacher_buffer.dtype),
                alpha=1.0 - ema_decay,
            )
        else:
            teacher_buffer.copy_(student_buffer)

    teacher.eval()


@torch.no_grad()
def select_dual_view_pseudo_labels(logits1, logits2, thresholds):
    """Strict two-view pseudo-label filtering.

    A target sample is reliable only when both weak views predict the same class
    and EACH view independently exceeds that class's adaptive threshold.
    """
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1)

    conf1, pred1 = probs1.max(dim=1)
    conf2, pred2 = probs2.max(dim=1)
    agreement = pred1.eq(pred2)

    pseudo_targets = pred1
    thresholds = thresholds.to(device=logits1.device, dtype=conf1.dtype)
    sample_threshold = thresholds.index_select(0, pseudo_targets)
    reliable = (
        agreement
        & (conf1 >= sample_threshold)
        & (conf2 >= sample_threshold)
    )

    return pseudo_targets, reliable.float(), int(agreement.sum().item())
