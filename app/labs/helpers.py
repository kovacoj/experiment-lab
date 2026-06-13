from __future__ import annotations

from app.labs.schemas import LabStatus


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def default_status(score: float, confidence: float) -> LabStatus:
    if score >= 0.70 and confidence >= 0.65:
        return "selected"
    if score >= 0.45 and confidence >= 0.50:
        return "warning"
    return "hidden"
