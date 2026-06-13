from __future__ import annotations

from typing import Literal

from app.text_engine.schemas import TextDocumentSignal


AdapterMode = Literal["deterministic", "mocked", "live"]


class TextModelAdapter:
    def __init__(self, mode: AdapterMode = "deterministic") -> None:
        self.mode = mode

    def classify(self, documents: list[TextDocumentSignal]) -> list[TextDocumentSignal]:
        if self.mode in {"deterministic", "mocked"}:
            return documents
        raise NotImplementedError("Live model inference is intentionally out of scope for this demo template.")
