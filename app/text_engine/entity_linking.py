from __future__ import annotations

from app.text_engine.schemas import TextDocumentSignal
from app.text_engine.taxonomies import ENTITY_ALIASES


def link_entities(documents: list[TextDocumentSignal]) -> list[TextDocumentSignal]:
    linked: list[TextDocumentSignal] = []
    for document in documents:
        aliases = ENTITY_ALIASES[document.scenario]
        entity_name = document.entity_name
        entity_type = document.entity_type
        if entity_name in aliases:
            entity_name, entity_type = aliases[entity_name]
        else:
            for alias, linked_entity in aliases.items():
                if alias.lower() in document.text.lower():
                    entity_name, entity_type = linked_entity
                    break
        linked.append(document.model_copy(update={"entity_name": entity_name, "entity_type": entity_type}))
    return linked
