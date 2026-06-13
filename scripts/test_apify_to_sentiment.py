from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.modeling.sentiment import LocalSentimentAnalyzer


TEXT_KEYS = ["text", "content", "description", "snippet", "markdown", "title"]

DOT_PREFIXES = ["searchResult", "metadata", "crawl", "page", "result"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for key in ["items", "results", "data", "output"]:
            value = raw.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [raw]
    return []


def _get_dot_key(item: dict[str, Any], key: str) -> Any:
    value = item.get(key)
    if value is not None:
        return value
    for prefix in DOT_PREFIXES:
        value = item.get(f"{prefix}.{key}")
        if value is not None:
            return value
    return None


def extract_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in TEXT_KEYS:
        value = _get_dot_key(item, key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    text = " ".join(parts)
    return " ".join(text.split())


def extract_url(item: dict[str, Any]) -> str | None:
    for key in ["url", "sourceUrl", "link", "pageUrl"]:
        value = _get_dot_key(item, key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    return None


def extract_title(item: dict[str, Any]) -> str | None:
    value = _get_dot_key(item, "title")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def normalize_docs(items: list[dict[str, Any]], scenario: str = "reputation_monitor") -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        text = extract_text(item)
        if not text:
            continue
        docs.append(
            {
                "document_id": f"apify_{scenario}_{idx}",
                "scenario": scenario,
                "source_type": "website",
                "source_name": "apify",
                "entity_name": item.get("entity_name") or "unknown",
                "entity_type": item.get("entity_type") or "unknown",
                "observed_at": item.get("date") or item.get("publishedAt") or item.get("crawl.loadedAt"),
                "url": extract_url(item),
                "title": extract_title(item),
                "text": text[:2000],
                "metadata": {
                    "raw_keys": sorted(list(item.keys())),
                    "raw_index": idx,
                },
            }
        )
    return docs


def sentiment_value(label: str) -> int:
    if label == "negative":
        return -1
    if label == "positive":
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--docs-out", default=None)
    parser.add_argument("--scenario", default="reputation_monitor")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_json(raw_path)
    items = flatten_items(raw)
    docs = normalize_docs(items, scenario=args.scenario)

    if args.docs_out:
        docs_path = Path(args.docs_out)
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")

    analyzer = LocalSentimentAnalyzer()
    predictions = analyzer.predict(
        texts=[doc["text"] for doc in docs],
        text_ids=[doc["document_id"] for doc in docs],
    )

    signals: list[dict[str, Any]] = []
    for doc, pred in zip(docs, predictions):
        signals.append(
            {
                "document_id": doc["document_id"],
                "scenario": doc["scenario"],
                "entity_name": doc["entity_name"],
                "entity_type": doc["entity_type"],
                "signal_type": "sentiment",
                "label": pred.label,
                "value": sentiment_value(pred.label),
                "confidence": pred.score,
                "evidence_text": doc["text"][:500],
                "observed_at": doc["observed_at"],
                "source_url": doc["url"],
                "metadata": {
                    "title": doc["title"],
                    "source_type": doc["source_type"],
                    "source_name": doc["source_name"],
                    "model_id": pred.model_id,
                    "model_mode": pred.model_mode,
                    "raw_label": pred.raw_label,
                },
            }
        )

    out_path.write_text(json.dumps(signals, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "raw_items": len(items),
                "normalized_docs": len(docs),
                "sentiment_signals": len(signals),
                "out": str(out_path),
                "docs_out": args.docs_out,
                "label_counts": {
                    label: sum(1 for s in signals if s["label"] == label)
                    for label in ["negative", "neutral", "positive"]
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
