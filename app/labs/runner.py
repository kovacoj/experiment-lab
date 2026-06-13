from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import polars as pl

from app.labs.base import BaseLab
from app.labs.critic import validate_lab_result
from app.labs.decision_cards import compile_decision_cards
from app.labs.registry import LAB_REGISTRY
from app.labs.selection import build_ensemble_findings, select_labs
from app.labs.schemas import DataSource, LabContext, LabResult, LabRunReport
from app.text_engine.cleaning import clean_documents
from app.text_engine.dedup import deduplicate_documents
from app.text_engine.entity_linking import link_entities
from app.text_engine.model_adapter import TextModelAdapter
from app.text_engine.rule_extractors import extract_text_signals
from app.text_engine.source_adapters import adapt_raw_records_to_documents, load_raw_external_records


pl.enable_string_cache()


def run_labs(context: LabContext, labs: list[BaseLab]) -> list[LabResult]:
    results: list[LabResult] = []
    for lab in labs:
        try:
            result = lab.run(context)
        except Exception as exc:
            result = LabResult(
                lab_id=lab.lab_id,
                lab_name=lab.lab_name,
                scenario=context.scenario,
                hypothesis="Lab failed before hypothesis could be evaluated.",
                status="failed",
                score=0.0,
                confidence=0.0,
                summary=f"Lab failed: {exc}",
                evidence=[],
                limitations=[str(exc)],
            )
        results.append(result)
    return results


def run_all_labs(scenario: str, context: LabContext, lab_ids: list[str] | None = None) -> list[LabResult]:
    context = prepare_context_for_labs(context)
    lab_classes = LAB_REGISTRY[scenario]
    if lab_ids is not None:
        allowed = set(lab_ids)
        lab_classes = [lab_class for lab_class in lab_classes if lab_class.lab_id in allowed]
    labs = [lab_class() for lab_class in lab_classes]
    raw_results = run_labs(context, labs)
    return [validate_lab_result(result, context) for result in raw_results]


def select_top_labs(results: list[LabResult], max_selected: int = 3, max_warning: int = 2) -> dict[str, list[LabResult]]:
    return select_labs(results, max_selected=max_selected, max_warning=max_warning)


def build_report(context: LabContext, results: list[LabResult]) -> LabRunReport:
    context = prepare_context_for_labs(context)
    grouped = select_labs(
        results,
        max_selected=int(context.analysis_contract.get("max_selected", 3)),
        max_warning=int(context.analysis_contract.get("max_warning", 2)),
    )
    ensembles = build_ensemble_findings(grouped["selected"] + grouped["warning"] + grouped["hidden"])
    executive_summary = _build_executive_summary(context.scenario, grouped["selected"], grouped["warning"], ensembles)
    return LabRunReport(
        scenario=context.scenario,
        selected=grouped["selected"],
        warning=grouped["warning"],
        hidden=grouped["hidden"],
        failed=grouped["failed"],
        discarded=grouped["discarded"],
        ensembles=ensembles,
        executive_summary=executive_summary,
    )


def load_demo_context(scenario: str) -> LabContext:
    root = Path(__file__).resolve().parents[1]
    demo_dir = root / "demo_data"
    return LabContext(
        scenario=scenario,
        internal_data={"path": demo_dir / f"{scenario}_internal.ndjson"},
        external_data={"path": demo_dir / f"{scenario}_external_raw.ndjson"},
        metadata={"demo_mode": True},
        analysis_contract={"max_selected": 3, "max_warning": 2},
    )


def list_labs(scenario: str) -> list[dict[str, str]]:
    return [
        {"lab_id": lab_class.lab_id, "lab_name": lab_class.lab_name}
        for lab_class in LAB_REGISTRY[scenario]
    ]


def run_demo_scenario(scenario: str, lab_ids: list[str] | None = None, benchmark: bool = False, rows: int = 100000) -> tuple[LabContext, LabRunReport]:
    context = build_benchmark_context(scenario, rows) if benchmark else load_demo_context(scenario)
    results = run_all_labs(scenario, context, lab_ids=lab_ids)
    report = build_report(context, results)
    return prepare_context_for_labs(context), report


def build_benchmark_context(scenario: str, rows: int) -> LabContext:
    base_context = load_demo_context(scenario)
    raw_records = _load_external_records(base_context.external_data)
    repeated_records: list[dict[str, object]] = []
    target_rows = max(rows, len(raw_records))
    for index in range(target_rows):
        source_record = dict(raw_records[index % len(raw_records)])
        source_record["document_id"] = f"{source_record['document_id']}-bench-{index}"
        repeated_records.append(source_record)

    return base_context.model_copy(
        update={
            "external_data": DataSource(records=repeated_records),
            "metadata": {
                **base_context.metadata,
                "benchmark_mode": True,
                "benchmark_rows": target_rows,
            },
        }
    )


def prepare_context_for_labs(context: LabContext) -> LabContext:
    if context.text_signals is not None and context.text_documents is not None:
        return context

    raw_records = _load_external_records(context.external_data)
    documents = adapt_raw_records_to_documents(context.scenario, raw_records)
    cleaned_documents = clean_documents(documents)
    deduped_documents = deduplicate_documents(cleaned_documents)
    linked_documents = link_entities(deduped_documents)
    classified_documents = TextModelAdapter(mode="deterministic").classify(linked_documents)
    signals = extract_text_signals(classified_documents)

    return context.model_copy(
        update={
            "text_documents": DataSource(records=[_document_to_record(document) for document in classified_documents]),
            "text_signals": DataSource(records=[_signal_to_record(signal) for signal in signals]),
            "metadata": {
                **context.metadata,
                "raw_external_document_count": len(raw_records),
                "text_document_count": len(classified_documents),
                "text_signal_count": len(signals),
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Signal Foundry demo labs")
    parser.add_argument("--scenario", required=True, choices=sorted(LAB_REGISTRY.keys()))
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--list-labs", action="store_true")
    parser.add_argument("--lab-id", action="append", default=[])
    args = parser.parse_args()

    if args.list_labs:
        labs = list_labs(args.scenario)
        if args.format == "json":
            print(json.dumps({"scenario": args.scenario, "labs": labs}, indent=2, sort_keys=True))
        else:
            print(render_lab_list(args.scenario, labs))
        return

    selected_lab_ids = args.lab_id or None
    if selected_lab_ids is not None:
        known_lab_ids = {lab["lab_id"] for lab in list_labs(args.scenario)}
        unknown_lab_ids = sorted(set(selected_lab_ids) - known_lab_ids)
        if unknown_lab_ids:
            raise SystemExit(f"Unknown lab ids for {args.scenario}: {', '.join(unknown_lab_ids)}")

    started_at = time.perf_counter()
    context, report = run_demo_scenario(args.scenario, lab_ids=selected_lab_ids, benchmark=args.benchmark, rows=args.rows)
    runtime_seconds = time.perf_counter() - started_at
    output = render_report(report) if args.format == "text" else render_report_json(report, context, runtime_seconds)
    print(output)


def render_report(report: LabRunReport) -> str:
    lines = [f"Scenario: {report.scenario}", ""]
    if report.ensembles:
        lines.append("Ensembles:")
        for ensemble in report.ensembles:
            lines.append(f"[{ensemble.final_priority_score:.2f}] {ensemble.title} - {ensemble.summary}")
        lines.append("")
    for section_name in ("selected", "warning", "hidden", "discarded", "failed"):
        lines.append(f"{section_name.capitalize()}:")
        section = getattr(report, section_name)
        if not section:
            lines.append("- none")
        else:
            for result in section:
                score = result.final_priority_score if result.final_priority_score is not None else result.score
                lines.append(f"[{score:.2f}] {result.lab_name} - {result.summary}")
        lines.append("")
    lines.append(f"Executive summary: {report.executive_summary}")
    return "\n".join(lines)


def render_lab_list(scenario: str, labs: list[dict[str, str]]) -> str:
    lines = [f"Scenario: {scenario}", "", "Available labs:"]
    for lab in labs:
        lines.append(f"- {lab['lab_id']}: {lab['lab_name']}")
    return "\n".join(lines)


def render_report_json(report: LabRunReport, context: LabContext, runtime_seconds: float | None = None) -> str:
    payload = {
        "scenario": report.scenario,
        "decision_cards": [card.model_dump(mode="json") for card in compile_decision_cards(report)],
        "ensembles": [ensemble.model_dump(mode="json") for ensemble in report.ensembles],
        "selected": [result.model_dump(mode="json") for result in report.selected],
        "warning": [result.model_dump(mode="json") for result in report.warning],
        "hidden": [result.model_dump(mode="json") for result in report.hidden],
        "discarded": [result.model_dump(mode="json") for result in report.discarded],
        "failed": [result.model_dump(mode="json") for result in report.failed],
        "executive_summary": report.executive_summary,
        "metadata": context.metadata,
    }
    if runtime_seconds is not None:
        payload["runtime_seconds"] = round(runtime_seconds, 4)
    return json.dumps(payload, indent=2, sort_keys=True)


def _build_executive_summary(scenario: str, selected: list[LabResult], warning: list[LabResult], ensembles) -> str:
    if ensembles:
        return " ".join(ensemble.summary for ensemble in ensembles[:2])
    primary = selected or warning
    if not primary:
        return f"No strong findings were selected for {scenario}."
    return " ".join(result.summary for result in primary[:3])


def _load_external_records(source: DataSource) -> list[dict[str, object]]:
    if source.path is not None:
        return load_raw_external_records(source.path)
    return list(source.records or [])


def _document_to_record(document) -> dict[str, object]:
    return {
        "dataset": "text_documents",
        "document_id": document.document_id,
        "scenario": document.scenario,
        "source_type": document.source_type,
        "source_name": document.source_name,
        "entity_name": document.entity_name,
        "entity_type": document.entity_type,
        "observed_at": document.observed_at.isoformat() if document.observed_at else None,
        "url": document.url,
        "title": document.title,
        "text": document.text,
        "language": document.language,
        "period": document.metadata.get("period"),
        "time_bucket": document.metadata.get("time_bucket"),
    }


def _signal_to_record(signal) -> dict[str, object]:
    return {
        "dataset": "text_signals",
        "document_id": signal.document_id,
        "scenario": signal.scenario,
        "source_type": signal.source_type,
        "source_name": signal.source_name,
        "entity_name": signal.entity_name,
        "entity_type": signal.entity_type,
        "signal_type": signal.signal_type,
        "label": signal.label,
        "value": signal.value,
        "numeric_value": signal.numeric_value,
        "confidence": signal.confidence,
        "evidence_text": signal.evidence_text,
        "observed_at": signal.observed_at.isoformat() if signal.observed_at else None,
        "period": signal.period,
        "time_bucket": signal.time_bucket,
    }


if __name__ == "__main__":
    main()
