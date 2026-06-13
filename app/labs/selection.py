from __future__ import annotations

from app.labs.schemas import EnsembleFinding, EvidenceItem, LabResult, RecommendedAction


def annotate_lab_utility(results: list[LabResult]) -> list[LabResult]:
    annotated: list[LabResult] = []
    duplicate_counts = _duplicate_theme_counts(results)
    for result in results:
        if result.status == "failed":
            annotated.append(
                result.model_copy(
                    update={
                        "signal_strength": 0.0,
                        "business_relevance": 0.0,
                        "actionability": 0.0,
                        "novelty": 0.0,
                        "data_quality_penalty": 0.0,
                        "duplication_penalty": 0.0,
                        "safety_penalty": 0.0,
                        "cost_penalty": 0.0,
                        "exploitation_score": 0.0,
                        "exploration_bonus": 0.0,
                        "final_priority_score": 0.0,
                        "reason_for_selection": "Lab failed before producing a usable result.",
                        "reason_for_hiding": None,
                        "reason_for_discarding": None,
                    }
                )
            )
            continue

        signal_strength = result.score
        business_relevance = _business_relevance(result)
        actionability = _actionability(result)
        novelty = _novelty(result)
        data_quality_penalty = min(len(result.limitations) * 0.03, 0.18)
        duplication_penalty = 0.08 if duplicate_counts[_theme_key(result)] > 1 and result.lab_id in {"staff_mention"} else 0.0
        safety_penalty = 0.35 if result.status == "discarded" else (0.10 if any("privacy" in item.lower() for item in result.limitations) else 0.0)
        cost_penalty = 0.04 if result.lab_id in {"competitor_price", "shipping_risk", "geopolitical"} else 0.02
        exploitation_score = (
            0.30 * signal_strength
            + 0.25 * result.confidence
            + 0.20 * business_relevance
            + 0.15 * actionability
            + 0.10 * novelty
            - data_quality_penalty
            - duplication_penalty
            - safety_penalty
            - cost_penalty
        )
        exploration_bonus = 0.05 if 0.30 <= result.score < 0.70 and novelty >= 0.60 and result.confidence < 0.70 else 0.0
        final_priority_score = exploitation_score + exploration_bonus
        reason = _reason_for_selection(result, final_priority_score)
        reason_for_hiding = _reason_for_hiding(result, final_priority_score)
        reason_for_discarding = _reason_for_discarding(result)
        annotated.append(
            result.model_copy(
                update={
                    "signal_strength": round(signal_strength, 4),
                    "business_relevance": round(business_relevance, 4),
                    "actionability": round(actionability, 4),
                    "novelty": round(novelty, 4),
                    "data_quality_penalty": round(data_quality_penalty, 4),
                    "duplication_penalty": round(duplication_penalty, 4),
                    "safety_penalty": round(safety_penalty, 4),
                    "cost_penalty": round(cost_penalty, 4),
                    "exploitation_score": round(exploitation_score, 4),
                    "exploration_bonus": round(exploration_bonus, 4),
                    "final_priority_score": round(final_priority_score, 4),
                    "reason_for_selection": reason,
                    "reason_for_hiding": reason_for_hiding,
                    "reason_for_discarding": reason_for_discarding,
                }
            )
        )
    return annotated


def select_labs(results: list[LabResult], max_selected: int = 3, max_warning: int = 2) -> dict[str, list[LabResult]]:
    annotated = annotate_lab_utility(results)
    selected: list[LabResult] = []
    warning: list[LabResult] = []
    hidden: list[LabResult] = []
    failed: list[LabResult] = []
    discarded: list[LabResult] = []

    for result in sorted(annotated, key=lambda item: item.final_priority_score or -1.0, reverse=True):
        if result.status == "failed":
            failed.append(result)
        elif result.status == "discarded":
            discarded.append(result)
        elif result.status == "selected":
            selected.append(result)
        elif result.status == "warning":
            warning.append(result)
        else:
            hidden.append(result)

    overflow_selected = selected[max_selected:]
    overflow_warning = warning[max_warning:]
    selected = selected[:max_selected]
    warning = warning[:max_warning]
    hidden.extend(
        item.model_copy(
            update={
                "status": "hidden",
                "reason_for_selection": item.reason_for_selection,
                "reason_for_hiding": "Valid result, but stronger labs consumed the available screen-space budget.",
            }
        )
        for item in overflow_selected + overflow_warning
    )
    hidden.sort(key=lambda item: item.final_priority_score or 0.0, reverse=True)

    return {
        "selected": selected,
        "warning": warning,
        "hidden": hidden,
        "failed": failed,
        "discarded": discarded,
    }


def build_ensemble_findings(results: list[LabResult]) -> list[EnsembleFinding]:
    by_id = {result.lab_id: result for result in results if result.status not in {"failed", "discarded"}}
    ensembles: list[EnsembleFinding] = []

    reputation_inputs = [by_id.get("location_sentiment"), by_id.get("peak_hours"), by_id.get("staff_mention")]
    reputation_inputs = [result for result in reputation_inputs if result is not None and result.status in {"selected", "warning", "hidden"}]
    if len(reputation_inputs) >= 2:
        ensembles.append(
            EnsembleFinding(
                ensemble_id="reputation_operational_risk",
                scenario="reputation_monitor",
                title="Operational Risk Card",
                summary="Service-quality signals point to an operational capacity issue across monitored Miners locations, with Vinohrady sentiment pressure and a Wenceslas morning peak overload.",
                status="selected",
                contributing_lab_ids=[result.lab_id for result in reputation_inputs],
                evidence=_take_evidence(reputation_inputs),
                recommended_actions=[
                    RecommendedAction(
                        title="Investigate service capacity",
                        detail="Prioritize morning service-capacity fixes in Vinohrady and Wenceslas, then monitor sentiment and queue recovery.",
                        urgency="high",
                    )
                ],
                confidence=round(sum(result.confidence for result in reputation_inputs) / len(reputation_inputs), 4),
                final_priority_score=round(sum((result.final_priority_score or 0.0) for result in reputation_inputs) / len(reputation_inputs), 4),
                reason_for_selection="Independent operational labs support the same service-capacity conclusion.",
            )
        )

    supply_inputs = [by_id.get("chip_supply"), by_id.get("shipping_risk"), by_id.get("production_stop_risk")]
    supply_inputs = [result for result in supply_inputs if result is not None and result.status in {"selected", "warning", "hidden"}]
    if len(supply_inputs) >= 2:
        ensembles.append(
            EnsembleFinding(
                ensemble_id="supply_chain_production_risk",
                scenario="supply_chain_risk",
                title="Production Risk Card",
                summary="Component lead-time pressure, logistics delay, and production-stop estimates combine into one near-term production-risk finding.",
                status="selected",
                contributing_lab_ids=[result.lab_id for result in supply_inputs],
                evidence=_take_evidence(supply_inputs),
                recommended_actions=[
                    RecommendedAction(
                        title="Protect production continuity",
                        detail="Expedite MCU supply, account for shipping delay, and protect the next production batch before the threshold window closes.",
                        urgency="critical",
                    )
                ],
                confidence=round(sum(result.confidence for result in supply_inputs) / len(supply_inputs), 4),
                final_priority_score=round(sum((result.final_priority_score or 0.0) for result in supply_inputs) / len(supply_inputs), 4),
                reason_for_selection="Independent risk labs support the same production-stop conclusion.",
            )
        )

    return ensembles


def _take_evidence(results: list[LabResult]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()
    for result in results:
        for item in result.evidence:
            key = (item.source, item.label, str(item.value))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
    return evidence[:6]


def _theme_key(result: LabResult) -> str:
    if result.lab_id in {"location_sentiment", "peak_hours", "staff_mention"}:
        return "reputation_operational"
    if result.lab_id in {"chip_supply", "shipping_risk", "production_stop_risk"}:
        return "supply_chain_production"
    return result.lab_id


def _duplicate_theme_counts(results: list[LabResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        key = _theme_key(result)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _business_relevance(result: LabResult) -> float:
    if result.lab_id in {"location_sentiment", "competitor_price", "chip_supply", "production_stop_risk", "battery_cost"}:
        return 0.90
    if result.lab_id in {"peak_hours", "shipping_risk", "alternative_supplier"}:
        return 0.78
    return 0.55


def _actionability(result: LabResult) -> float:
    if not result.recommended_actions:
        return 0.30
    urgency_weights = {"low": 0.45, "medium": 0.65, "high": 0.82, "critical": 0.95}
    return max(urgency_weights[action.urgency] for action in result.recommended_actions)


def _novelty(result: LabResult) -> float:
    if result.lab_id in {"menu_trend", "staff_mention", "geopolitical", "alternative_supplier"}:
        return 0.72
    if result.lab_id in {"competitor_price", "shipping_risk"}:
        return 0.62
    return 0.50


def _reason_for_selection(result: LabResult, final_priority_score: float) -> str:
    if result.status == "selected":
        return f"Strong actionable signal with priority score {final_priority_score:.2f}."
    if result.status == "warning":
        return f"Relevant finding kept visible despite limitations; priority score {final_priority_score:.2f}."
    if result.status == "hidden":
        return f"Valid finding evaluated with priority score {final_priority_score:.2f}."
    if result.status == "discarded":
        return "Result was evaluated but not promoted into user-facing reasoning."
    return "Result was not promotable beyond audit/debug output."


def _reason_for_hiding(result: LabResult, final_priority_score: float) -> str | None:
    if result.status == "hidden":
        return f"Valid but weak or non-urgent finding; priority score {final_priority_score:.2f}."
    return None


def _reason_for_discarding(result: LabResult) -> str | None:
    if result.status == "discarded":
        if any("privacy" in item.lower() or "causality" in item.lower() for item in result.limitations):
            return "Discarded because the finding was unsafe for user-facing reasoning."
        if any("unsupported" in item.lower() or "no evidence" in item.lower() for item in result.limitations):
            return "Discarded because the finding was unsupported by evidence."
        return "Discarded because the finding was invalid, unsafe, or irrelevant to the analysis contract."
    return None
