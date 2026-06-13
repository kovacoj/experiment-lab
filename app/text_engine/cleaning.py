from __future__ import annotations

import re

from app.text_engine.schemas import TextDocumentSignal


WHITESPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"<[^>]+>")


def clean_documents(documents: list[TextDocumentSignal]) -> list[TextDocumentSignal]:
    cleaned: list[TextDocumentSignal] = []
    for document in documents:
        text = WHITESPACE_RE.sub(" ", HTML_RE.sub(" ", document.text)).strip()
        title = None if document.title is None else WHITESPACE_RE.sub(" ", HTML_RE.sub(" ", document.title)).strip()
        cleaned.append(document.model_copy(update={"text": text, "title": title}))
    return cleaned
