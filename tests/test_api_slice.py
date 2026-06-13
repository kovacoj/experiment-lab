"""End-to-end test for the FastAPI slice.

Exercises the full happy path the n8n MCP workflow calls into:
  refresh -> monitoring-plan -> dashboard -> chart data
plus the explicit /monitoring-plan/build endpoint and a 404 for an
unknown session.

Runs against the real lab pipeline (no mocking) but redirects on-disk
state into a per-test tmp dir via EXPERIMENT_LAB_API_TMP_DIR.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("EXPERIMENT_LAB_API_TMP_DIR", str(tmp_path))
    # Reimport so storage.base_dir() picks up the patched env at call time.
    # (base_dir reads the env on every call, so reload is defensive only.)
    from app.api import main as api_main

    importlib.reload(api_main)
    return TestClient(api_main.app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get("/sessions/demo_unknown/monitoring-plan")
    assert response.status_code == 404


def test_refresh_then_dashboard_then_chart_data(client: TestClient) -> None:
    session_id = "demo_miners"

    # Refresh runs the labs, builds the plan, and appends one chart point.
    refresh_response = client.post(f"/sessions/{session_id}/refresh")
    assert refresh_response.status_code == 200
    refresh_payload = refresh_response.json()
    assert refresh_payload["session_id"] == session_id
    assert refresh_payload["scenario"] == "reputation_monitor"
    assert refresh_payload["prediction_changed"] is True
    # Alert envelope is now derived from real lab results: Miners Vinohrady
    # sits at sentiment ~0.55 (critical) in the cached fixture, so the
    # refresh now emits should_notify=true with severity in {warning, critical}.
    assert refresh_payload["alert"]["should_notify"] is True
    assert refresh_payload["alert"]["severity"] in {"critical", "warning"}
    assert "sentiment_trend_by_location" in refresh_payload["dashboard_delta"]["updated_chart_ids"]

    # Monitoring plan is persisted and discoverable.
    plan_response = client.get(f"/sessions/{session_id}/monitoring-plan")
    assert plan_response.status_code == 200
    plan = plan_response.json()
    assert plan["session_id"] == session_id
    assert plan["scenario"] == "reputation_monitor"
    model_ids = [model["model_id"] for model in plan["monitored_models"]]
    assert "location_sentiment_drop_v1" in model_ids
    sentiment_model = next(m for m in plan["monitored_models"] if m["model_id"] == "location_sentiment_drop_v1")
    assert "sentiment_trend_by_location" in sentiment_model["chart_ids"]

    # Dashboard exposes the chart with at least one series.
    dashboard_response = client.get(f"/sessions/{session_id}/dashboard")
    assert dashboard_response.status_code == 200
    dashboard = dashboard_response.json()
    chart_ids = [chart["chart_id"] for chart in dashboard["charts"]]
    assert "sentiment_trend_by_location" in chart_ids
    sentiment_chart = next(c for c in dashboard["charts"] if c["chart_id"] == "sentiment_trend_by_location")
    assert sentiment_chart["x_key"] == "time"
    assert len(sentiment_chart["series"]) > 0
    series_keys = {series["key"] for series in sentiment_chart["series"]}

    # Chart data endpoint returns the appended history points.
    data_response = client.get(f"/sessions/{session_id}/charts/sentiment_trend_by_location/data")
    assert data_response.status_code == 200
    data = data_response.json()
    assert data["chart_id"] == "sentiment_trend_by_location"
    assert len(data["data"]) == 1
    point = data["data"][0]
    assert "time" in point
    # Every chart-declared series should have a value in the appended point.
    for key in series_keys:
        assert key in point, f"chart point missing series {key}"

    # A second refresh appends a new point — chart data grows over time.
    second = client.post(f"/sessions/{session_id}/refresh")
    assert second.status_code == 200
    data2 = client.get(f"/sessions/{session_id}/charts/sentiment_trend_by_location/data").json()
    assert len(data2["data"]) == 2


def test_build_only_endpoint_persists_plan_without_history(client: TestClient, tmp_path) -> None:
    session_id = "demo_miners"
    response = client.post(f"/sessions/{session_id}/monitoring-plan/build")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert any(model["model_id"] == "location_sentiment_drop_v1" for model in body["monitored_models"])

    # Chart history must NOT be advanced by /build.
    data = client.get(f"/sessions/{session_id}/charts/sentiment_trend_by_location/data").json()
    assert data["data"] == []


def test_unknown_chart_id_returns_404(client: TestClient) -> None:
    client.post("/sessions/demo_miners/refresh")
    response = client.get("/sessions/demo_miners/charts/no_such_chart/data")
    assert response.status_code == 404


def test_bundle_rebuild_writes_expected_files(client: TestClient, tmp_path) -> None:
    """POST /sessions/{id}/bundle/rebuild regenerates the static dashboard bundle.

    The bundle lives at base_dir().parent / "bundle" so a single
    EXPERIMENT_LAB_API_TMP_DIR override relocates it for tests.
    """
    session_id = "demo_miners"
    response = client.post(f"/sessions/{session_id}/bundle/rebuild")
    assert response.status_code == 200
    body = response.json()
    assert body["bundle_dir"].endswith("/_bundle")
    expected = {
        "dashboard_payload.json",
        "chart_specs.json",
        "finding_cards.json",
        "explanation_cards.json",
        "prediction_payload.json",
        "reports.json",
        "seed_metadata.json",
        "lovable_prompt.md",
    }
    assert expected.issubset(set(body["files"]))

    bundle_dir = tmp_path / "_bundle"
    for name in expected:
        assert (bundle_dir / name).exists(), f"missing bundle file: {name}"

    # The bundle is served as a static mount, so the dashboard_payload.json
    # must be retrievable over HTTP at /bundle/<name>.
    served = client.get("/bundle/dashboard_payload.json")
    assert served.status_code == 200
    payload = served.json()
    assert "business" in payload
    assert "kpis" in payload
    assert "locations" in payload


def test_root_redirects_to_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


# ---------------------------------------------------------------------------
# Sales forecast endpoint


def test_sales_forecast_default_horizon(client: TestClient) -> None:
    response = client.get("/sessions/demo_miners/forecasts/sales")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "demo_miners"
    assert body["scenario"] == "reputation_monitor"
    assert body["horizon_days"] == 30
    assert len(body["daily"]) == 30


def test_sales_forecast_custom_horizon_and_payload_shape(client: TestClient) -> None:
    response = client.get("/sessions/demo_miners/forecasts/sales?horizon_days=14")
    assert response.status_code == 200
    body = response.json()

    # Horizon honored.
    assert body["horizon_days"] == 14
    assert len(body["daily"]) == 14

    # Every daily entry has the headline fields and a factor decomposition.
    required = {
        "date",
        "day_of_week",
        "baseline_revenue_eur",
        "predicted_revenue_eur",
        "predicted_with_intervention_eur",
        "p10_revenue_eur",
        "p90_revenue_eur",
        "expected_orders",
        "expected_avg_order_eur",
        "factors",
        "per_location",
    }
    for entry in body["daily"]:
        assert required.issubset(entry), f"missing keys on {entry['date']}"
        # P10 <= predicted <= P90 by construction.
        assert entry["p10_revenue_eur"] <= entry["predicted_revenue_eur"] <= entry["p90_revenue_eur"]
        # Per-location rollup must cover all four Miners locations.
        assert set(entry["per_location"].keys()) == {
            "Miners Vinohrady",
            "Miners Wenceslas",
            "Miners Letna",
            "Miners Karlin",
        }

    # Scenarios include the three demo scenarios and report a non-negative uplift
    # for the staffing action (intervention closes the sentiment-drop gap).
    scenario_ids = [s["scenario_id"] for s in body["scenarios"]]
    assert scenario_ids == ["do_nothing", "add_morning_staff", "competitive_promo"]
    staff = next(s for s in body["scenarios"] if s["scenario_id"] == "add_morning_staff")
    do_nothing = next(s for s in body["scenarios"] if s["scenario_id"] == "do_nothing")
    assert staff["vs_baseline_eur"] >= do_nothing["vs_baseline_eur"]

    # Feature importance is a valid probability-ish distribution (~sums to 1).
    importance_sum = sum(f["importance"] for f in body["feature_importance"])
    assert 0.95 <= importance_sum <= 1.05

    # Per-location next-7d view covers every location.
    assert {row["location_name"] for row in body["by_location_next_7d"]} == {
        "Miners Vinohrady",
        "Miners Wenceslas",
        "Miners Letna",
        "Miners Karlin",
    }

    # Model metadata declares an identifier and MAPE.
    meta = body["model_metadata"]
    assert meta["model_id"] == "revenue_forecast_v1"
    assert "validation_mape" in meta
    assert "features" in meta and "day_of_week" in meta["features"]


def test_sales_forecast_clamps_horizon(client: TestClient) -> None:
    too_long = client.get("/sessions/demo_miners/forecasts/sales?horizon_days=500")
    assert too_long.status_code == 200
    assert too_long.json()["horizon_days"] == 90

    too_short = client.get("/sessions/demo_miners/forecasts/sales?horizon_days=1")
    assert too_short.status_code == 200
    assert too_short.json()["horizon_days"] == 7


def test_sales_forecast_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get("/sessions/demo_unknown/forecasts/sales")
    assert response.status_code == 404


def test_sales_forecast_is_deterministic_per_scenario(client: TestClient) -> None:
    """Two back-to-back calls produce identical forecast values (timestamps
    excepted), which lets the dashboard cache it safely between refreshes."""
    a = client.get("/sessions/demo_miners/forecasts/sales?horizon_days=21").json()
    b = client.get("/sessions/demo_miners/forecasts/sales?horizon_days=21").json()
    assert a["daily"] == b["daily"]
    assert a["totals"] == b["totals"]
    assert a["scenarios"] == b["scenarios"]


# ---------------------------------------------------------------------------
# Chat endpoint (stub keyword-routed responder)

def test_chat_routes_status_intent_and_returns_grounded_reply(client: TestClient) -> None:
    response = client.post(
        "/sessions/demo_miners/chat",
        json={"message": "what's the status?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "status"
    assert payload["engine"] == "stub-keyword-router-v1"
    assert "Snapshot" in payload["reply"]
    assert payload["data"]["weakest"]  # display name pulled from real snapshot
    assert any(s["ref"] == "ui:forecasts" for s in payload["sources"])


def test_chat_forecast_intent_parses_horizon_from_message(client: TestClient) -> None:
    response = client.post(
        "/sessions/demo_miners/chat",
        json={"message": "give me the 60-day forecast"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "forecast"
    assert payload["data"]["horizon_days"] == 60
    assert "60-day revenue forecast" in payload["reply"]


def test_chat_unknown_intent_falls_back_with_help_pointer(client: TestClient) -> None:
    response = client.post(
        "/sessions/demo_miners/chat",
        json={"message": "asdfqwer nothing matches here"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "unknown"
    assert "help" in payload["reply"].lower()


def test_chat_rejects_empty_message_with_400(client: TestClient) -> None:
    response = client.post(
        "/sessions/demo_miners/chat",
        json={"message": "   "},
    )
    assert response.status_code == 400


def test_chat_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post(
        "/sessions/demo_unknown/chat",
        json={"message": "hello"},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Cards + alerts (n8n MCP sf_get_session_state / sf_get_session_alerts)

def test_cards_404_before_first_refresh(client: TestClient) -> None:
    response = client.get("/sessions/demo_miners/cards")
    assert response.status_code == 404


def test_refresh_populates_cards_and_alerts_endpoints(client: TestClient) -> None:
    refresh = client.post("/sessions/demo_miners/refresh")
    assert refresh.status_code == 200
    body = refresh.json()
    # Refresh response now carries cards + a real alert envelope
    assert body["decision_cards"], "refresh should populate decision_cards"
    assert body["alert"]["should_notify"] in (True, False)

    cards = client.get("/sessions/demo_miners/cards")
    assert cards.status_code == 200
    cards_payload = cards.json()
    assert cards_payload["session_id"] == "demo_miners"
    assert cards_payload["scenario"] == "reputation_monitor"
    assert isinstance(cards_payload["cards"], list) and cards_payload["cards"]
    # Card shape: must carry the fields the dashboard / MCP consumer needs
    first = cards_payload["cards"][0]
    for key in ("card_id", "title", "card_type", "summary", "priority"):
        assert key in first

    alerts = client.get("/sessions/demo_miners/alerts")
    assert alerts.status_code == 200
    alerts_payload = alerts.json()
    assert alerts_payload["session_id"] == "demo_miners"
    assert "alerts" in alerts_payload
    assert alerts_payload["count"] == len(alerts_payload["alerts"])
    for entry in alerts_payload["alerts"]:
        assert entry["severity"] in {"critical", "warning"}
        assert entry["lab_id"]
        assert entry["created_at"]


def test_alerts_appends_across_refreshes(client: TestClient) -> None:
    client.post("/sessions/demo_miners/refresh")
    first = client.get("/sessions/demo_miners/alerts").json()
    client.post("/sessions/demo_miners/refresh")
    second = client.get("/sessions/demo_miners/alerts").json()
    # Second refresh must not shrink the alert log
    assert second["count"] >= first["count"]


def test_cards_and_alerts_unknown_session_404(client: TestClient) -> None:
    assert client.get("/sessions/demo_unknown/cards").status_code == 404
    assert client.get("/sessions/demo_unknown/alerts").status_code == 404
