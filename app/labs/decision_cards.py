from __future__ import annotations

from app.labs.schemas import DecisionCard, EnsembleFinding, LabResult, LabRunReport


def compile_decision_cards(report: LabRunReport) -> list[DecisionCard]:
    cards: list[DecisionCard] = []

    for ensemble in report.ensembles:
        cards.append(_ensemble_to_card(ensemble))

    ensembled_lab_ids = {lab_id for ensemble in report.ensembles for lab_id in ensemble.contributing_lab_ids}

    for result in report.selected:
        if result.lab_id in ensembled_lab_ids and _is_covered_by_ensemble(result, report.ensembles):
            continue
        cards.append(_lab_to_card(result, card_type="finding"))

    for result in report.warning:
        if result.lab_id in ensembled_lab_ids and _is_covered_by_ensemble(result, report.ensembles):
            continue
        cards.append(_lab_to_card(result, card_type="warning"))

    cards.sort(key=lambda card: card.priority, reverse=True)
    return cards


def _ensemble_to_card(ensemble: EnsembleFinding) -> DecisionCard:
    return DecisionCard(
        card_id=ensemble.ensemble_id,
        scenario=ensemble.scenario,
        title=ensemble.title,
        card_type="ensemble",
        summary=ensemble.summary,
        priority=ensemble.final_priority_score,
        status=ensemble.status,
        supporting_lab_ids=ensemble.contributing_lab_ids,
        evidence=ensemble.evidence,
        recommended_actions=ensemble.recommended_actions,
        caveats=[],
    )


def _lab_to_card(result: LabResult, *, card_type: str) -> DecisionCard:
    caveats = list(result.limitations)
    if result.reason_for_hiding and result.status == "hidden":
        caveats.append(result.reason_for_hiding)
    return DecisionCard(
        card_id=f"card-{result.lab_id}",
        scenario=result.scenario,
        title=result.lab_name,
        card_type=card_type,
        summary=result.summary,
        priority=result.final_priority_score or result.score,
        status="warning" if result.status == "warning" else "selected",
        supporting_lab_ids=[result.lab_id],
        evidence=result.evidence,
        recommended_actions=result.recommended_actions,
        caveats=caveats,
    )


def _is_covered_by_ensemble(result: LabResult, ensembles: list[EnsembleFinding]) -> bool:
    return any(result.lab_id in ensemble.contributing_lab_ids for ensemble in ensembles)
