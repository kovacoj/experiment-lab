from __future__ import annotations

import unittest
from pathlib import Path

import polars as pl

from app.labs.runner import load_demo_context, prepare_context_for_labs
from app.text_engine.cleaning import clean_documents
from app.text_engine.dedup import deduplicate_documents
from app.text_engine.entity_linking import link_entities
from app.text_engine.rule_extractors import extract_text_signals
from app.text_engine.source_adapters import adapt_raw_records_to_documents, load_raw_external_records


class TextEngineTests(unittest.TestCase):
    def test_raw_external_records_normalize_into_text_documents(self) -> None:
        raw_path = Path("/home/cady/personal/kaggle/signal_foundry_template/app/demo_data/reputation_monitor_external_raw.ndjson")
        raw_records = load_raw_external_records(raw_path)
        documents = adapt_raw_records_to_documents("reputation_monitor", raw_records)
        self.assertGreater(len(documents), 0)
        self.assertEqual(documents[0].scenario, "reputation_monitor")

    def test_text_documents_become_extracted_signals(self) -> None:
        raw_path = Path("/home/cady/personal/kaggle/signal_foundry_template/app/demo_data/supply_chain_risk_external_raw.ndjson")
        raw_records = load_raw_external_records(raw_path)
        documents = adapt_raw_records_to_documents("supply_chain_risk", raw_records)
        linked_documents = link_entities(deduplicate_documents(clean_documents(documents)))
        signals = extract_text_signals(linked_documents)
        labels = {signal.label for signal in signals}
        self.assertIn("lead_time_increase", labels)
        self.assertIn("shipping_delay", labels)

    def test_reputation_demo_context_prepares_text_documents_and_signals(self) -> None:
        context = prepare_context_for_labs(load_demo_context("reputation_monitor"))
        self.assertGreater(context.metadata["text_document_count"], 0)
        self.assertGreater(context.metadata["text_signal_count"], 0)
        sentiment_count = (
            context.scan_text_signals(columns=["signal_type"])
            .filter(pl.col("signal_type") == "sentiment")
            .select(pl.len())
            .collect()
            .item(0, 0)
        )
        self.assertGreater(sentiment_count, 0)

    def test_supply_chain_demo_context_prepares_shared_text_signals(self) -> None:
        context = prepare_context_for_labs(load_demo_context("supply_chain_risk"))
        labels = set(
            context.scan_text_signals(columns=["label"])
            .collect()
            .get_column("label")
            .to_list()
        )
        self.assertIn("lead_time_increase", labels)
        self.assertIn("shipping_delay", labels)
        self.assertIn("geopolitical_risk", labels)


if __name__ == "__main__":
    unittest.main()
