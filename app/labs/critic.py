from __future__ import annotations

import re

from app.labs.helpers import default_status
from app.labs.schemas import LabContext, LabResult


EMPLOYEE_BLAME_PATTERN = re.compile(r"\b[A-Z][a-z]+ caused\b")


def validate_lab_result(result: LabResult, global_context: LabContext) -> LabResult:
    validated = result.model_copy(deep=True)

    if not validated.evidence and validated.status not in {"failed", "inconclusive", "discarded"}:
        validated.status = "discarded"
        validated.score = min(validated.score, 0.30)
        validated.confidence = min(validated.confidence, 0.30)
        validated.limitations.append("The lab returned no evidence, so the conclusion was discarded as unsupported.")

    if validated.status == "selected" and validated.confidence < 0.50:
        validated.status = "warning"
        validated.limitations.append("Confidence is below 0.50, so the result was downgraded.")

    if validated.status in {"selected", "warning", "hidden"}:
        threshold_status = default_status(validated.score, validated.confidence)
        if threshold_status == "warning" and validated.status == "selected" and validated.confidence < 0.65:
            validated.status = "warning"
            validated.limitations.append("Confidence is below the selected threshold.")
        elif threshold_status == "hidden" and validated.status in {"selected", "warning"}:
            validated.status = "hidden"
            validated.limitations.append("Signal strength is below the display threshold.")

    if _contains_employee_blame(validated):
        validated.status = "discarded"
        validated.summary = "A staff-related correlation was observed, but person-level attribution is not supported."
        validated.limitations.append("Employee-level blame was removed to preserve privacy and avoid overclaiming causality.")

    if _external_record_count(global_context) < 3:
        validated.limitations.append("External data coverage is thin, so external claims should be treated cautiously.")

    return validated


def _contains_employee_blame(result: LabResult) -> bool:
    texts = [result.summary]
    texts.extend(action.detail for action in result.recommended_actions)
    texts.extend(item.detail or "" for item in result.evidence)
    return any(EMPLOYEE_BLAME_PATTERN.search(text) for text in texts)


def _external_record_count(context: LabContext) -> int:
    return context.external_data.count_rows()
