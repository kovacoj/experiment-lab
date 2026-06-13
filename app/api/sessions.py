"""Session id → scenario registry.

The lab engine has no session concept; sessions live in the API layer only.
Only `demo_miners` is wired for the current slice. `demo_drones` is reserved
but not implemented.
"""
from __future__ import annotations

from typing import Literal, TypedDict

ScenarioName = Literal["reputation_monitor", "supply_chain_risk"]


class SessionInfo(TypedDict):
    session_id: str
    scenario: ScenarioName
    title: str


_SESSIONS: dict[str, SessionInfo] = {
    "demo_miners": {
        "session_id": "demo_miners",
        "scenario": "reputation_monitor",
        "title": "Miners cafés",
    },
}


def get_session(session_id: str) -> SessionInfo:
    info = _SESSIONS.get(session_id)
    if info is None:
        raise KeyError(session_id)
    return info


def known_session_ids() -> list[str]:
    return list(_SESSIONS.keys())
