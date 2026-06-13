from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class MenuTrendLab(BaseLab):
    lab_id = "menu_trend"
    lab_name = "Menu Trend Lab"
    scenario = "reputation_monitor"

    def run(self, context: LabContext) -> LabResult:
        trend_summary = (
            context.scan_text_signals(columns=["entity_name", "label", "signal_type", "period"])
            .filter((pl.col("signal_type") == "menu_trend") & (pl.col("label") == "oat_milk_trend"))
            .group_by("period")
            .agg(pl.len().alias("mention_count"))
            .collect()
        )
        recent_count = int(trend_summary.filter(pl.col("period") == "recent").item(0, "mention_count")) if trend_summary.filter(pl.col("period") == "recent").height else 0
        baseline_count = int(trend_summary.filter(pl.col("period") == "baseline").item(0, "mention_count")) if trend_summary.filter(pl.col("period") == "baseline").height else 0
        recent_growth = recent_count - baseline_count

        score = clamp(0.35 + min(max(recent_growth, 0) / 20.0, 0.10))
        confidence = 0.58
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="A rising menu trend can reveal an opportunity, but only if the signal is strong enough.",
            status=status,
            score=score,
            confidence=confidence,
            summary="Oat milk interest is rising in cached demo mentions, but the signal is still weak and should stay in monitoring only.",
            evidence=[
                EvidenceItem(source="external", label="oat_milk_growth", value=recent_growth, detail="Oat milk mentions increased slightly across reviews and social mentions."),
                EvidenceItem(source="derived", label="signal_strength", value=round(score, 2), detail="Trend strength is below the main finding threshold."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Keep monitoring menu trend",
                    detail="No immediate action; continue monitoring oat milk and related menu trends.",
                    urgency="low",
                )
            ],
            limitations=["Trend detection uses small cached demo mention counts rather than a broad social corpus."],
            monitoring_rules=[{"type": "trend_mentions", "trend": "oat milk", "threshold": 10}],
        )
