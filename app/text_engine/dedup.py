from __future__ import annotations

from hashlib import sha1

import polars as pl

from app.text_engine.schemas import TextDocumentSignal


def deduplicate_documents(documents: list[TextDocumentSignal]) -> list[TextDocumentSignal]:
    if not documents:
        return []
    rows = []
    lookup = {document.document_id: document for document in documents}
    for document in documents:
        normalized_hash = sha1(
            f"{document.url}|{document.entity_name}|{document.observed_at}|{document.text.lower()}".encode("utf-8")
        ).hexdigest()
        rows.append({"document_id": document.document_id, "dedup_key": normalized_hash})
    keep_ids = (
        pl.DataFrame(rows)
        .unique(subset=["dedup_key"], keep="first")
        .get_column("document_id")
        .to_list()
    )
    return [lookup[document_id] for document_id in keep_ids]
