from __future__ import annotations

import numpy as np


def display_probability(p: float | None, *, debug: bool = False) -> str:
    if p is None:
        return "—"
    try:
        value = float(p)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    value = float(np.clip(value, 0.0, 1.0))
    if debug:
        return f"{value:.6f}"
    if value >= 0.995:
        return "≥99%"
    if value <= 0.005:
        return "≤1%"
    return f"{value:.0%}"
