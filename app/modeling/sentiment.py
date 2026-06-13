from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import pipeline


MODEL_DIR = Path(".models/sentiment/cardiffnlp-twitter-xlm-roberta-base-sentiment")


@dataclass
class SentimentPrediction:
    document_id: str
    label: str
    raw_label: str
    score: float
    model_id: str
    model_mode: str


class LocalSentimentAnalyzer:
    def __init__(self, model_dir: Path | str | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self._pipeline = pipeline(
            task="sentiment-analysis",
            model=str(self._model_dir),
            tokenizer=str(self._model_dir),
            device=-1,
            truncation=True,
        )
        self._model_id = self._model_dir.name

    def predict(
        self,
        texts: list[str],
        text_ids: list[str] | None = None,
    ) -> list[SentimentPrediction]:
        if text_ids is None:
            text_ids = [str(i) for i in range(len(texts))]
        raw_results = self._pipeline(texts, batch_size=8)
        predictions: list[SentimentPrediction] = []
        for text_id, raw in zip(text_ids, raw_results):
            raw_label = raw["label"].lower()
            label = _normalize_label(raw_label)
            predictions.append(
                SentimentPrediction(
                    document_id=text_id,
                    label=label,
                    raw_label=raw_label,
                    score=raw["score"],
                    model_id=self._model_id,
                    model_mode="offline",
                )
            )
        return predictions


def _normalize_label(raw_label: str) -> str:
    if "positive" in raw_label:
        return "positive"
    if "negative" in raw_label:
        return "negative"
    return "neutral"
