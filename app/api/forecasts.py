"""Sales / revenue forecasts for the dashboard.

Deterministic synthetic forecast generator. Pulls per-location sentiment from
the prepared lab context so the forecast story stays consistent with the
rest of the dashboard (sentiment drop at Miners Vinohrady drags predicted
demand at that location, the staffing-action scenario closes the gap).

This is a presentation-layer module; it does NOT add a new lab. The
analytics labs remain pure analytics. The forecast is layered on top of the
lab context for demo purposes only.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from app.api.sentiment_metrics import location_sentiment_snapshot, series_key
from app.labs.runner import load_demo_context, prepare_context_for_labs

# Per-location daily baseline revenue (EUR) for the Miners chain demo.
_LOCATION_BASELINE_EUR: dict[str, float] = {
    "Miners Vinohrady": 3400.0,
    "Miners Wenceslas": 4100.0,  # tourist-heavy, highest baseline
    "Miners Letna":     2900.0,
    "Miners Karlin":    3200.0,
}

# Mon..Sun seasonality multipliers; office-heavy mid-week peak, weekend dip.
_DOW_SEASONALITY: list[float] = [1.08, 1.10, 1.07, 1.05, 1.06, 0.96, 0.78]

# Czech public holidays in the near horizon. Deterministic for demo.
_HOLIDAYS: dict[str, str] = {
    "2026-07-05": "Saints Cyril and Methodius",
    "2026-07-06": "Jan Hus Day",
    "2026-09-28": "St. Wenceslas Day",
    "2026-10-28": "Independent Czechoslovak State Day",
}

_MAX_HORIZON_DAYS = 90
_MIN_HORIZON_DAYS = 7
_DEFAULT_HORIZON_DAYS = 30


def build_sales_forecast(
    session_id: str,
    scenario: str,
    horizon_days: int = _DEFAULT_HORIZON_DAYS,
    start_date: str | None = None,
) -> dict[str, Any]:
    """Synthesize a deterministic sales-forecast payload.

    The forecast values are stable per scenario — only `generated_at` and
    `trained_at` change between calls. Horizon is clamped to
    [7, 90] days. `start_date` (YYYY-MM-DD) is optional; defaults to today UTC.
    """
    horizon_days = max(_MIN_HORIZON_DAYS, min(int(horizon_days), _MAX_HORIZON_DAYS))

    raw_ctx = load_demo_context(scenario)
    ctx = prepare_context_for_labs(raw_ctx)
    sentiment = location_sentiment_snapshot(ctx)

    rng = random.Random(_seed_for(scenario))

    # Sentiment-based per-location revenue adjustment.
    # Centered at 0.75 ("normal"). 1.0 → +6%, 0.5 → -18% (linear, capped).
    location_adjust: dict[str, float] = {}
    for name in _LOCATION_BASELINE_EUR:
        s = sentiment.get(series_key(name), 0.75)
        location_adjust[name] = (s - 0.75) * 0.96

    start = (
        datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        if start_date
        else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    )

    daily: list[dict[str, Any]] = []
    by_location_7d: dict[str, dict[str, float]] = {}

    total_baseline_pool = sum(_LOCATION_BASELINE_EUR.values())

    for i in range(horizon_days):
        date = start + timedelta(days=i)
        dow_idx = date.weekday()
        dow_mult = _DOW_SEASONALITY[dow_idx]
        date_str = date.strftime("%Y-%m-%d")
        is_holiday = date_str in _HOLIDAYS
        holiday_mult = 0.55 if is_holiday else 1.0

        per_loc: dict[str, dict[str, float]] = {}
        total_baseline = 0.0
        total_predicted = 0.0
        total_with_intervention = 0.0

        for name, base in _LOCATION_BASELINE_EUR.items():
            jitter = 1.0 + (rng.random() - 0.5) * 0.04  # ±2% noise
            sent_mult = 1.0 + location_adjust[name]
            competitor_drag = 0.96 if name == "Miners Vinohrady" else 1.0

            baseline_rev = base * dow_mult * holiday_mult * jitter
            predicted_rev = baseline_rev * sent_mult * competitor_drag

            # Intervention only affects Vinohrady; recovers to baseline over ~5 days.
            recovery = min(1.0, (i + 1) / 5.0) if name == "Miners Vinohrady" else 0.0
            with_intervention = predicted_rev + (baseline_rev - predicted_rev) * recovery

            per_loc[name] = {
                "baseline_eur": _round_eur(baseline_rev),
                "predicted_eur": _round_eur(predicted_rev),
                "with_intervention_eur": _round_eur(with_intervention),
            }
            total_baseline += baseline_rev
            total_predicted += predicted_rev
            total_with_intervention += with_intervention

            if i < 7:
                acc = by_location_7d.setdefault(name, {"baseline_eur": 0.0, "predicted_eur": 0.0})
                acc["baseline_eur"] += baseline_rev
                acc["predicted_eur"] += predicted_rev

        # Confidence interval grows slightly with horizon distance.
        ci_width = 0.06 + 0.001 * i
        p10 = total_predicted * (1.0 - ci_width)
        p90 = total_predicted * (1.0 + ci_width)

        avg_order_eur = 17.4 + rng.uniform(-0.4, 0.4)
        expected_orders = round(total_predicted / avg_order_eur)

        # Per-day factor decomposition (€ delta vs baseline).
        vinohrady_base_today = _LOCATION_BASELINE_EUR["Miners Vinohrady"] * dow_mult
        factors = [
            {"name": "weekly_seasonality",         "contribution_eur": _round_eur((dow_mult - 1.0) * total_baseline_pool)},
            {"name": "sentiment_drop_vinohrady",   "contribution_eur": _round_eur(location_adjust["Miners Vinohrady"] * vinohrady_base_today)},
            {"name": "competitor_promo",           "contribution_eur": _round_eur(-0.04 * vinohrady_base_today)},
            {"name": "holiday_effect",             "contribution_eur": _round_eur((holiday_mult - 1.0) * total_baseline_pool * dow_mult)},
        ]

        daily.append(
            {
                "date": date_str,
                "day_of_week": date.strftime("%a"),
                "is_holiday": is_holiday,
                "holiday_name": _HOLIDAYS.get(date_str),
                "baseline_revenue_eur": _round_eur(total_baseline),
                "predicted_revenue_eur": _round_eur(total_predicted),
                "predicted_with_intervention_eur": _round_eur(total_with_intervention),
                "p10_revenue_eur": _round_eur(p10),
                "p90_revenue_eur": _round_eur(p90),
                "expected_orders": expected_orders,
                "expected_avg_order_eur": round(avg_order_eur, 2),
                "factors": factors,
                "per_location": per_loc,
            }
        )

    tot_baseline = sum(d["baseline_revenue_eur"] for d in daily)
    tot_predicted = sum(d["predicted_revenue_eur"] for d in daily)
    tot_with = sum(d["predicted_with_intervention_eur"] for d in daily)

    by_location_view: list[dict[str, Any]] = []
    for name in sorted(by_location_7d.keys()):
        v = by_location_7d[name]
        vs_baseline = ((v["predicted_eur"] - v["baseline_eur"]) / v["baseline_eur"]) if v["baseline_eur"] else 0.0
        by_location_view.append(
            {
                "location_name": name,
                "next_7d_baseline_eur": _round_eur(v["baseline_eur"]),
                "next_7d_predicted_eur": _round_eur(v["predicted_eur"]),
                "vs_baseline_pct": round(vs_baseline, 4),
            }
        )

    intervention_cost = 4200  # one extra morning-shift staff for 5 days
    intervention_uplift = tot_with - tot_predicted
    intervention_net = intervention_uplift - intervention_cost

    anomalies: list[dict[str, Any]] = []
    for d in daily:
        if d["is_holiday"]:
            anomalies.append(
                {
                    "date": d["date"],
                    "type": "expected_drop",
                    "reason": f"{d['holiday_name']} — reduced footfall predicted",
                    "delta_eur": _round_eur(d["predicted_revenue_eur"] - d["baseline_revenue_eur"]),
                }
            )

    worst = min(daily, key=lambda d: d["predicted_revenue_eur"] - d["baseline_revenue_eur"])
    if worst and not worst["is_holiday"]:
        anomalies.append(
            {
                "date": worst["date"],
                "type": "deepest_dip",
                "reason": "Largest negative deviation from baseline driven by sentiment-drop persistence.",
                "delta_eur": _round_eur(worst["predicted_revenue_eur"] - worst["baseline_revenue_eur"]),
            }
        )

    now = datetime.now(timezone.utc).replace(microsecond=0)
    vinohrady_drop_pct = abs(round(location_adjust["Miners Vinohrady"] * 100))

    return {
        "session_id": session_id,
        "scenario": scenario,
        "generated_at": now.isoformat(),
        "horizon_days": horizon_days,
        "method": (
            "Additive seasonal-trend decomposition with intervention overlay "
            "(weekly seasonality + sentiment-driven demand model)"
        ),
        "model_metadata": {
            "model_id": "revenue_forecast_v1",
            "trained_on_days": 90,
            "validation_mae_eur": 612,
            "validation_mape": 0.054,
            "validation_smape": 0.051,
            "trained_at": (now - timedelta(hours=18)).isoformat(),
            "next_retrain_at": (now + timedelta(hours=6)).isoformat(),
            "features": [
                "day_of_week",
                "holiday_flag",
                "weather_proxy",
                "sentiment_score_rolling_7d",
                "staffing_index",
                "competitor_promo_flag",
                "lagged_revenue_7d",
                "lagged_revenue_28d",
            ],
        },
        "totals": {
            "baseline_horizon_revenue_eur": _round_eur(tot_baseline),
            "predicted_horizon_revenue_eur": _round_eur(tot_predicted),
            "predicted_with_intervention_eur": _round_eur(tot_with),
            "do_nothing_gap_eur": _round_eur(tot_predicted - tot_baseline),
            "intervention_uplift_eur": _round_eur(intervention_uplift),
            "intervention_uplift_pct": round(intervention_uplift / tot_predicted, 4) if tot_predicted else 0.0,
            "intervention_cost_eur": intervention_cost,
            "intervention_net_eur": _round_eur(intervention_net),
        },
        "scenarios": [
            {
                "scenario_id": "do_nothing",
                "label": "Do nothing",
                "horizon_revenue_eur": _round_eur(tot_predicted),
                "vs_baseline_eur": _round_eur(tot_predicted - tot_baseline),
                "vs_baseline_pct": round((tot_predicted - tot_baseline) / tot_baseline, 4) if tot_baseline else 0.0,
                "probability": 0.62,
                "cost_eur": 0,
                "net_eur": _round_eur(tot_predicted - tot_baseline),
            },
            {
                "scenario_id": "add_morning_staff",
                "label": "Add morning-shift staff at Miners Vinohrady (5 days)",
                "horizon_revenue_eur": _round_eur(tot_with),
                "vs_baseline_eur": _round_eur(tot_with - tot_baseline),
                "vs_baseline_pct": round((tot_with - tot_baseline) / tot_baseline, 4) if tot_baseline else 0.0,
                "probability": 0.84,
                "cost_eur": intervention_cost,
                "net_eur": _round_eur(intervention_net),
            },
            {
                "scenario_id": "competitive_promo",
                "label": "Launch 10% loyalty discount (counter-promo)",
                "horizon_revenue_eur": _round_eur(tot_predicted + intervention_uplift * 0.55),
                "vs_baseline_eur": _round_eur(tot_predicted + intervention_uplift * 0.55 - tot_baseline),
                "vs_baseline_pct": round(
                    (tot_predicted + intervention_uplift * 0.55 - tot_baseline) / tot_baseline, 4
                ) if tot_baseline else 0.0,
                "probability": 0.58,
                "cost_eur": 8200,
                "net_eur": _round_eur(intervention_uplift * 0.55 - 8200),
            },
        ],
        "by_location_next_7d": by_location_view,
        "feature_importance": [
            {"feature": "day_of_week",                "importance": 0.42},
            {"feature": "sentiment_score_rolling_7d", "importance": 0.21},
            {"feature": "weather_proxy",              "importance": 0.14},
            {"feature": "staffing_index",             "importance": 0.11},
            {"feature": "competitor_promo_flag",      "importance": 0.07},
            {"feature": "holiday_flag",               "importance": 0.05},
        ],
        "narrative": (
            f"Forecast horizon: {horizon_days} days. Dominant signal is weekly seasonality "
            f"(orders peak Mon–Wed). The Miners Vinohrady sentiment drop reduces predicted "
            f"demand at that location by ~{vinohrady_drop_pct}% across the horizon unless an "
            f"intervention is taken. The 'add morning staff' scenario closes the gap in ~5 "
            f"days for an expected net contribution of €{int(intervention_net):,}."
        ),
        "daily": daily,
        "anomalies": anomalies,
        # Operator-supplied notes ride along so the dashboard can render
        # them alongside the projection. Populated by the API layer
        # (forecasts.py is pure / deterministic and does not read disk),
        # default None means "no operator context attached".
        "user_context": None,
    }


def _seed_for(scenario: str) -> int:
    return sum(ord(c) for c in scenario) * 17 + 42


def _round_eur(value: float) -> int:
    return int(round(value))
