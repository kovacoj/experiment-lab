from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.labs.decision_cards import compile_decision_cards
from app.labs.schemas import DecisionCard
from app.labs.registry import LAB_REGISTRY
from app.labs.runner import (
    build_report,
    list_labs,
    load_demo_context,
    render_report,
    run_all_labs,
    run_demo_scenario,
)


SUPPORTED_SCENARIOS = sorted(LAB_REGISTRY.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal Foundry experiment-lab skill CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-scenarios")

    list_labs_parser = subparsers.add_parser("list-labs")
    list_labs_parser.add_argument("--scenario", required=True)
    list_labs_parser.add_argument("--format", choices=["json", "text"], default="json")

    run_analysis_parser = subparsers.add_parser("run-analysis")
    run_analysis_parser.add_argument("--scenario", required=True)
    run_analysis_parser.add_argument("--format", choices=["json", "text"], default="json")

    refresh_compare_parser = subparsers.add_parser("refresh-compare")
    refresh_compare_parser.add_argument("--scenario", required=True)
    refresh_compare_parser.add_argument("--format", choices=["json", "text"], default="json")

    args = parser.parse_args()

    try:
        if args.command == "list-scenarios":
            result = {"status": "ok", "scenarios": SUPPORTED_SCENARIOS}
            _emit(result, output_format="json")
            return

        scenario = args.scenario
        _validate_scenario(scenario)

        if args.command == "list-labs":
            result = {
                "scenario": scenario,
                "status": "ok",
                "labs": list_labs(scenario),
            }
            _emit(result, output_format=args.format)
            return

        if args.command == "run-analysis":
            result = run_analysis(scenario)
            _emit(result, output_format=args.format)
            return

        if args.command == "refresh-compare":
            result = refresh_compare(scenario)
            _emit(result, output_format=args.format)
            return

        raise ValueError(f"Unsupported command: {args.command}")

    except Exception as exc:  # pragma: no cover - exercised through CLI tests
        scenario = getattr(args, "scenario", None)
        error = {
            "scenario": scenario,
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        print(json.dumps(error, indent=2, ensure_ascii=False))
        sys.exit(1)


def run_analysis(scenario: str) -> dict[str, Any]:
    context, report = run_demo_scenario(scenario)
    decision_cards = compile_decision_cards(report)
    return {
        "scenario": scenario,
        "status": "ok",
        "analysis_contract": context.analysis_contract,
        "lab_run_report": report.model_dump(mode="json"),
        "decision_cards": [card.model_dump(mode="json") for card in decision_cards],
        "alert_candidates": _alert_candidates(decision_cards),
        "metadata": {
            "data_mode": "demo_fixture",
            "external_mode": "cached",
            "hf_mode": "off",
            **context.metadata,
        },
    }


def refresh_compare(scenario: str) -> dict[str, Any]:
    old_result = run_analysis(scenario)
    new_result = run_analysis(scenario)
    old_cards = old_result["decision_cards"]
    new_cards = new_result["decision_cards"]
    old_headline, old_confidence = _headline_and_confidence(old_cards, old_result["lab_run_report"]["executive_summary"])
    new_headline, new_confidence = _headline_and_confidence(new_cards, new_result["lab_run_report"]["executive_summary"])
    prediction_changed = old_headline != new_headline or old_confidence != new_confidence
    alert = None
    if prediction_changed:
        alert = {
            "title": new_headline,
            "confidence": new_confidence,
            "reason": "Top decision card or confidence changed after refresh.",
        }
    return {
        "scenario": scenario,
        "status": "ok",
        "prediction_changed": prediction_changed,
        "old_headline": old_headline,
        "new_headline": new_headline,
        "old_confidence": old_confidence,
        "new_confidence": new_confidence,
        "alert": alert,
        "metadata": {
            "data_mode": "demo_fixture",
            "external_mode": "cached",
            "hf_mode": "off",
        },
    }


def _alert_candidates(decision_cards: list[DecisionCard]) -> list[dict[str, Any]]:
    return [
        {
            "card_id": card.card_id,
            "title": card.title,
            "summary": card.summary,
            "priority": card.priority,
            "status": card.status,
        }
        for card in decision_cards
        if card.status in {"selected", "warning"} and card.priority >= 0.5
    ]


def _headline_and_confidence(decision_cards: list[dict[str, Any]], executive_summary: str) -> tuple[str, float]:
    if decision_cards:
        top_card = max(decision_cards, key=lambda card: float(card["priority"]))
        return top_card["summary"], float(top_card["priority"])
    return executive_summary, 0.0


def _validate_scenario(scenario: str) -> None:
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")


def _emit(result: dict[str, Any], output_format: str) -> None:
    if output_format == "text":
        if result.get("status") != "ok":
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        if "labs" in result:
            print(_render_text_labs(result))
            return
        if "lab_run_report" in result:
            print(_render_text_analysis(result))
            return
        if "prediction_changed" in result:
            print(_render_text_refresh(result))
            return
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _render_text_labs(result: dict[str, Any]) -> str:
    lines = [f"Scenario: {result['scenario']}", "", "Available labs:"]
    for lab in result["labs"]:
        lines.append(f"- {lab['lab_id']}: {lab['lab_name']}")
    return "\n".join(lines)


def _render_text_analysis(result: dict[str, Any]) -> str:
    report = result["lab_run_report"]
    context = load_demo_context(result["scenario"])
    lab_results = run_all_labs(result["scenario"], context)
    report_obj = build_report(context, lab_results)
    return render_report(report_obj)


def _render_text_refresh(result: dict[str, Any]) -> str:
    lines = [f"Scenario: {result['scenario']}", "", f"Prediction changed: {result['prediction_changed']}"]
    lines.append(f"Old headline: {result['old_headline']}")
    lines.append(f"New headline: {result['new_headline']}")
    lines.append(f"Old confidence: {result['old_confidence']:.4f}")
    lines.append(f"New confidence: {result['new_confidence']:.4f}")
    if result["alert"]:
        lines.append(f"Alert: {result['alert']['title']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
