"""Mask-only candidate routing; no truth labels or training-state changes."""


def uniform_radius_candidates(confidence, agreement, radius_pass, enabled):
    direct = confidence & agreement
    empty = direct & ~direct
    if not enabled:
        return direct, empty, empty
    accepted_confidence = direct & radius_pass
    rescued = agreement & ~confidence & radius_pass
    rejected_confidence = direct & ~radius_pass
    return accepted_confidence, rescued, rejected_confidence
