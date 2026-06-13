"""Stub chat synthesizer for the "0 to 100" dashboard.

This is not an LLM. It is a deterministic, intent-routed responder that
pulls real numbers from the current lab context + sales forecast so the
chat window on the dashboard feels grounded.

Design goals:
    * No external network calls — labs-first contract holds.
    * Pure function of (message, session_id, scenario); no global mutation.
    * Replies always cite the dashboard panel or endpoint they came from
      so the user can sanity-check a number against the UI.

Intent routing is keyword based. The first matching intent wins.
"""
from __future__ import annotations

import re
from typing import Any

from app.api.forecasts import build_sales_forecast
from app.api.sentiment_metrics import location_sentiment_snapshot
from app.labs.runner import load_demo_context, prepare_context_for_labs

# ---------------------------------------------------------------------------
# Intent keyword tables. Order matters: first match wins.

_INTENTS: list[tuple[str, list[str]]] = [
    ("help", ["help", "what can", "how do i", "what do you", "examples"]),
    ("status", ["status", "summary", "overview", "headline", "alert", "what changed", "what's new"]),
    ("locations", ["location", "vinohrady", "wenceslas", "letna", "karlin", "branch", "store"]),
    ("sentiment", ["sentiment", "review", "mood", "complaint", "rating", "score"]),
    ("forecast", ["forecast", "revenue", "sales", "predict", "next month", "horizon", "eur", "€"]),
    ("intervention", [
        "intervention", "fix", "recommend", "recommendation", "action",
        "what should", "what would you", "morning staff", "promo",
    ]),
    ("model", ["model", "mape", "mae", "feature importance", "training", "retrain"]),
    ("anomaly", ["anomaly", "anomalies", "holiday", "spike", "dip", "outlier"]),
]


def _detect_intent(message: str) -> str:
    text = message.lower()
    for intent, keywords in _INTENTS:
        for kw in keywords:
            if kw in text:
                return intent
    return "unknown"


def _eur(value: float | int) -> str:
    return f"€{int(round(value)):,}".replace(",", " ")


def _pct(value: float, *, sign: bool = True) -> str:
    fmt = "+" if sign and value > 0 else ""
    return f"{fmt}{value * 100:.1f}%"


def _display_name(slug: str) -> str:
    """`miners_vinohrady_sentiment` → `Miners Vinohrady`."""
    parts = [p for p in slug.split("_") if p and p != "sentiment"]
    return " ".join(p.capitalize() for p in parts)


def _location_key(slug: str) -> str:
    """`miners_vinohrady_sentiment` → `vinohrady` (the geographic token)."""
    parts = [p for p in slug.split("_") if p and p not in {"miners", "sentiment"}]
    return parts[-1] if parts else slug


# ---------------------------------------------------------------------------
# Intent handlers. Each returns (reply_markdown, sources, data).

def _handle_help(_msg: str, _session: str, _scenario: str) -> tuple[str, list[dict], dict]:
    reply = (
        "I'm the **0 to 100** assistant. I answer questions grounded in the "
        "current dashboard data. I'm a stub, not a real LLM — replies are "
        "keyword routed and always show the source panel.\n\n"
        "**Try asking:**\n"
        "- *What's the current status?*\n"
        "- *How is Vinohrady doing?*\n"
        "- *What does the 30-day revenue forecast say?*\n"
        "- *What intervention do you recommend?*\n"
        "- *Tell me about the forecast model.*\n"
        "- *Are there any anomalies coming up?*"
    )
    return reply, [{"label": "Dashboard overview", "ref": "ui:overview"}], {}


def _handle_status(_msg: str, session_id: str, scenario: str) -> tuple[str, list[dict], dict]:
    context = prepare_context_for_labs(load_demo_context(scenario))
    snapshot = location_sentiment_snapshot(context)
    if not snapshot:
        return ("No sentiment snapshot available yet — run /refresh first.", [], {})
    weakest_slug, weakest_score = min(snapshot.items(), key=lambda kv: kv[1])
    strongest_slug, strongest_score = max(snapshot.items(), key=lambda kv: kv[1])
    avg = sum(snapshot.values()) / len(snapshot)
    fc = build_sales_forecast(session_id=session_id, scenario=scenario, horizon_days=30)
    gap = fc["totals"]["do_nothing_gap_eur"]
    baseline = fc["totals"]["baseline_horizon_revenue_eur"]

    reply = (
        f"**Snapshot for `{session_id}` ({scenario}):**\n"
        f"- Network sentiment avg: **{avg:.2f}** ({len(snapshot)} locations)\n"
        f"- Weakest: **{_display_name(weakest_slug)}** at {weakest_score:.2f}\n"
        f"- Strongest: **{_display_name(strongest_slug)}** at {strongest_score:.2f}\n"
        f"- 30-day revenue gap vs baseline (do nothing): **{_eur(gap)}** "
        f"({_pct(gap / baseline)})\n"
        f"- Recommended intervention net: **{_eur(fc['totals']['intervention_net_eur'])}**"
    )
    return reply, [
        {"label": "Sentiment by location", "ref": "ui:overview"},
        {"label": "Sales forecast", "ref": "ui:forecasts"},
    ], {
        "avg_sentiment": round(avg, 3),
        "weakest": _display_name(weakest_slug),
        "gap_eur": gap,
    }


def _handle_locations(message: str, _session: str, scenario: str) -> tuple[str, list[dict], dict]:
    context = prepare_context_for_labs(load_demo_context(scenario))
    snapshot = location_sentiment_snapshot(context)
    if not snapshot:
        return ("No sentiment snapshot available yet — run /refresh first.", [], {})
    text = message.lower()
    targets = [slug for slug in snapshot if _location_key(slug) in text]
    if targets:
        lines = []
        for slug in targets:
            score = snapshot[slug]
            verdict = "healthy" if score >= 0.75 else ("watch" if score >= 0.6 else "at risk")
            lines.append(f"- **{_display_name(slug)}** — sentiment {score:.2f} ({verdict})")
        reply = "**Requested locations:**\n" + "\n".join(lines)
    else:
        ranked = sorted(snapshot.items(), key=lambda kv: kv[1])
        lines = [f"- **{_display_name(slug)}** — {score:.2f}" for slug, score in ranked]
        reply = "**Locations ranked from weakest to strongest:**\n" + "\n".join(lines)
    return reply, [{"label": "Sentiment by location", "ref": "ui:overview"}], {
        "snapshot": {_display_name(slug): score for slug, score in snapshot.items()},
    }


def _handle_sentiment(_msg: str, _session: str, scenario: str) -> tuple[str, list[dict], dict]:
    context = prepare_context_for_labs(load_demo_context(scenario))
    snapshot = location_sentiment_snapshot(context)
    if not snapshot:
        return ("No sentiment snapshot available yet — run /refresh first.", [], {})
    weakest_slug, weakest_score = min(snapshot.items(), key=lambda kv: kv[1])
    avg = sum(snapshot.values()) / len(snapshot)
    reply = (
        f"Network-wide sentiment is **{avg:.2f}**. The driver of any current "
        f"alert is **{_display_name(weakest_slug)}** at **{weakest_score:.2f}** — "
        f"that drop is what bends the predicted revenue line on the forecast tab.\n\n"
        f"Reviews source: cached demo evidence under `reputation_monitor` "
        f"scenario (see Findings tab for verbatim quotes)."
    )
    return reply, [
        {"label": "Sentiment trend chart", "ref": "ui:overview"},
        {"label": "Findings", "ref": "ui:findings"},
    ], {
        "avg": avg,
        "weakest_location": _display_name(weakest_slug),
        "weakest_score": weakest_score,
    }


def _handle_forecast(message: str, session_id: str, scenario: str) -> tuple[str, list[dict], dict]:
    horizon = 30
    m = re.search(r"(\d{1,3})\s*[-\s]?\s*(day|d|days)\b", message.lower())
    if m:
        horizon = max(7, min(90, int(m.group(1))))
    fc = build_sales_forecast(session_id=session_id, scenario=scenario, horizon_days=horizon)
    totals = fc["totals"]
    baseline = totals["baseline_horizon_revenue_eur"]
    gap = totals["do_nothing_gap_eur"]
    reply = (
        f"**{horizon}-day revenue forecast** for `{session_id}`:\n"
        f"- Baseline: **{_eur(baseline)}**\n"
        f"- Predicted (do nothing): **{_eur(totals['predicted_horizon_revenue_eur'])}** "
        f"({_pct(gap / baseline)} vs baseline)\n"
        f"- With recommended intervention: **{_eur(totals['predicted_with_intervention_eur'])}**\n"
        f"- Intervention uplift: **{_eur(totals['intervention_uplift_eur'])}** gross, "
        f"**{_eur(totals['intervention_net_eur'])}** net of cost\n\n"
        f"Method: {fc['method']}"
    )
    return reply, [{"label": "Sales forecast", "ref": "ui:forecasts"}], {
        "horizon_days": horizon,
        "totals": totals,
    }


def _handle_intervention(_msg: str, session_id: str, scenario: str) -> tuple[str, list[dict], dict]:
    fc = build_sales_forecast(session_id=session_id, scenario=scenario, horizon_days=30)
    scenarios = sorted(fc["scenarios"], key=lambda s: s["net_eur"], reverse=True)
    best = scenarios[0]
    lines = []
    for s in scenarios:
        lines.append(
            f"- **{s['label']}** — revenue {_eur(s['horizon_revenue_eur'])}, "
            f"cost {_eur(s['cost_eur'])}, net **{_eur(s['net_eur'])}** "
            f"(p={s.get('probability', 0):.2f})"
        )
    reply = (
        f"**Recommended next action: {best['label']}** (net {_eur(best['net_eur'])} over 30 days).\n\n"
        "**All scenarios compared:**\n" + "\n".join(lines) + "\n\n"
        "*Confidence reflects the model's probability of beating the do-nothing baseline.*"
    )
    return reply, [{"label": "Scenario comparison", "ref": "ui:forecasts"}], {
        "best_scenario_id": best["scenario_id"],
        "best_net_eur": best["net_eur"],
    }


def _handle_model(_msg: str, session_id: str, scenario: str) -> tuple[str, list[dict], dict]:
    fc = build_sales_forecast(session_id=session_id, scenario=scenario, horizon_days=30)
    meta = fc["model_metadata"]
    top_features = sorted(fc["feature_importance"], key=lambda f: f["importance"], reverse=True)[:3]
    feat_lines = "\n".join(
        f"  - **{f['feature']}** — {f['importance'] * 100:.0f}%" for f in top_features
    )
    reply = (
        f"**Forecast model: `{meta['model_id']}`**\n"
        f"- Validation MAPE: **{meta['validation_mape']:.1%}** · "
        f"MAE: **{_eur(meta['validation_mae_eur'])}**\n"
        f"- Trained on **{meta['trained_on_days']} days** of history\n"
        f"- Next retrain: **{meta['next_retrain_at']}**\n"
        f"- Features in use: **{len(meta['features'])}**\n\n"
        f"**Top features by importance:**\n{feat_lines}"
    )
    return reply, [{"label": "Forecast model card", "ref": "ui:forecasts"}], {
        "model_id": meta["model_id"],
    }


def _handle_anomaly(_msg: str, session_id: str, scenario: str) -> tuple[str, list[dict], dict]:
    fc = build_sales_forecast(session_id=session_id, scenario=scenario, horizon_days=30)
    anomalies = fc.get("anomalies", [])
    if not anomalies:
        return (
            "No anomalies flagged in the next 30 days.",
            [{"label": "Anomalies panel", "ref": "ui:forecasts"}],
            {},
        )
    lines = [
        f"- **{a['date']}** — {a['type'].replace('_', ' ')}: {a['reason']} "
        f"({_eur(a['delta_eur'])})"
        for a in anomalies
    ]
    reply = (
        f"**{len(anomalies)} anomaly point(s) flagged in the next 30 days:**\n"
        + "\n".join(lines)
    )
    return reply, [{"label": "Anomalies panel", "ref": "ui:forecasts"}], {
        "count": len(anomalies),
    }


def _handle_unknown(message: str, _session: str, _scenario: str) -> tuple[str, list[dict], dict]:
    reply = (
        f"I didn't catch a known topic in *“{message.strip()[:80]}”*. "
        "I'm a stub — try keywords like **status**, **forecast**, "
        "**vinohrady**, **intervention**, **model**, or **anomalies**, "
        "or type *help* for the full menu."
    )
    return reply, [{"label": "Help", "ref": "chat:help"}], {}


_HANDLERS = {
    "help": _handle_help,
    "status": _handle_status,
    "locations": _handle_locations,
    "sentiment": _handle_sentiment,
    "forecast": _handle_forecast,
    "intervention": _handle_intervention,
    "model": _handle_model,
    "anomaly": _handle_anomaly,
    "unknown": _handle_unknown,
}


def synthesize_reply(message: str, *, session_id: str, scenario: str) -> dict[str, Any]:
    """Route the message to a handler and return a chat response payload.

    Output shape:
        {
            "intent": "forecast",
            "reply": "**30-day revenue forecast** ...",     # markdown
            "sources": [{"label": "...", "ref": "..."}],
            "data": {...},                                  # intent-specific scraps
            "engine": "stub-keyword-router-v1",
        }
    """
    intent = _detect_intent(message)
    handler = _HANDLERS[intent]
    reply, sources, data = handler(message, session_id, scenario)
    return {
        "intent": intent,
        "reply": reply,
        "sources": sources,
        "data": data,
        "engine": "stub-keyword-router-v1",
    }
