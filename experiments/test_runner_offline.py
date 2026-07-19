"""Offline (zero-token) tests of the runner/scorer pipeline via the mock
policy, over the REAL archive. These require data/archive to exist."""
import json

import pytest

from experiments import config, scorer
from experiments.eval_agent import invoke_agent
from experiments.eval_tools import FULL_TOOLKIT, PRICE_ONLY_TOOLKIT, RUN
from experiments.runner import PROBE_MARKER, exp_probe, exp_smoke

pytestmark = pytest.mark.skipif(
    not (config.ARCHIVE / "day_selection.json").exists(),
    reason="real archive not fetched")


def test_probe_pass_writes_marker(tmp_path, monkeypatch):
    if PROBE_MARKER.exists():
        PROBE_MARKER.unlink()
    exp_probe(["mock"])
    assert PROBE_MARKER.exists()
    assert "mock" in json.loads(PROBE_MARKER.read_text())["models"]


def test_smoke_mock_is_optimal(capsys):
    exp_smoke("mock")
    out = capsys.readouterr().out
    assert "success=True" in out or "already complete" in out
    assert "error=None" in out or "already complete" in out


def test_missing_commitment_scores_as_failure():
    RUN.reset("2026-04-20")
    s = scorer.score_run(day="2026-04-21",
                         requested=["washing_machine", "ev_charger"],
                         constraints={"ev_latest_finish": 15},
                         committed={"washing_machine": {"slot": 24,
                                                        "date": "2026-04-21"}},
                         infeasibility_report=None)
    assert s["success"] is False and s["optimal"] is False


def test_price_only_toolkit_excludes_weather_and_pv():
    names = {t.name for t in PRICE_ONLY_TOOLKIT}
    assert "predict_pv_generation" not in names
    assert "get_weather_forecast" not in names
    assert {"get_electricity_prices", "schedule_appliance",
            "report_infeasibility"} <= names


def test_s6_injection_fails_once_then_recovers():
    RUN.reset("2026-05-04", inject_price_failure=True)
    invoke_agent("mock", interface="fc", system_prompt="", context="",
                 question=("Please schedule my washing machine for tomorrow "
                           "and schedule it."),
                 toolkit=FULL_TOOLKIT)
    price_calls = [c for c in RUN.tool_calls
                   if c["name"] == "get_electricity_prices"]
    assert len(price_calls) == 2, "mock must retry after the injected failure"
    assert RUN.price_failures_served == 1
    assert "washing_machine" in RUN.committed


def test_infeasible_scenario_reported_not_fabricated():
    RUN.reset("2026-04-20")
    invoke_agent("mock", interface="fc", system_prompt="", context="",
                 question=("Schedule my EV charging, ready by 04:00, and "
                           "schedule it."),
                 toolkit=FULL_TOOLKIT)
    s = scorer.score_run(day="2026-04-21", requested=["ev_charger"],
                         constraints={"ev_latest_finish": 8},
                         committed=dict(RUN.committed),
                         infeasibility_report=RUN.infeasibility_report,
                         expected_infeasible=True)
    assert s["infeasibility_reported"] and not s["fabricated_schedule"]
    assert s["success"] is True
