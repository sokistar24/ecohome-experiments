"""
M1 gate tests for the MILP optimizer (paper Appendix A).

T1  Decomposition equivalence: with no PV, no base load, and no power cap,
    the joint MILP must reproduce the anchor paper's independent
    per-appliance cheapest-window search, in both starts and objective.
T2  Negative-price complementarity: on an Agile-like day with negative
    import prices and exportable PV, the solution must never import and
    export in the same slot, and the MILP objective must equal the
    Eq. (1) evaluation of its own schedule (no relaxation exploit).
T3  S4 infeasibility: a deadline tighter than the EV duration must return
    status 'infeasible', never a schedule.
T4  PV shift (sanity): under flat prices, adding midday PV must pull the
    washing machine into the generation window and strictly reduce net
    cost versus the same schedule priced without PV awareness.
"""
import random

import pytest

from experiments import config
from experiments.optimizer import (ApplianceTask, cheapest_window,
                                   evaluate_schedule, solve)

T = config.SLOTS_PER_DAY


def standard_tasks(ev_latest_finish=None):
    a = config.APPLIANCES
    return [
        ApplianceTask("washing_machine", a["washing_machine"]["power_kw"],
                      a["washing_machine"]["slots"]),
        ApplianceTask("dishwasher", a["dishwasher"]["power_kw"],
                      a["dishwasher"]["slots"]),
        ApplianceTask("ev_charger", a["ev_charger"]["power_kw"],
                      a["ev_charger"]["slots"], latest_finish=ev_latest_finish),
    ]


def agile_like_prices(seed=7, negative=False):
    """Synthetic half-hourly curve: cheap overnight, morning+evening peaks."""
    rng = random.Random(seed)
    prices = []
    for t in range(T):
        h = t / 2
        base = 0.12
        if 6.5 <= h < 9.5:
            base = 0.32
        elif 16 <= h < 19.5:
            base = 0.28
        elif h < 5:
            base = 0.08
        prices.append(round(base + rng.uniform(-0.015, 0.015), 4))
    if negative:  # high-wind small hours, as Agile occasionally produces
        for t in range(4, 9):
            prices[t] = round(-0.03 - 0.01 * rng.random(), 4)
    return prices


def test_T1_decomposition_equivalence():
    prices = agile_like_prices()
    tasks = standard_tasks()
    res = solve(prices, tasks)  # no pv, no cap, B=0
    assert res["status"] == "optimal"
    expected_obj = 0.0
    for a in tasks:
        w = cheapest_window(prices, a.slots)
        window_cost = a.power_kw * config.SLOT_HOURS * w["window_sum"]
        expected_obj += window_cost
        joint = sum(prices[res["starts"][a.name]:res["starts"][a.name] + a.slots])
        # Starts may differ only under exact ties; window cost must match.
        assert joint == pytest.approx(w["window_sum"], abs=1e-9)
    assert res["objective"] == pytest.approx(expected_obj, abs=1e-6)


def test_T2_negative_price_complementarity():
    prices = agile_like_prices(negative=True)
    pv = [0.0] * T
    for t in range(18, 34):  # 09:00-17:00 generation, peak ~2.4 kWh/slot
        pv[t] = round(2.4 * (1 - abs(t - 26) / 8), 3)
    tasks = standard_tasks()
    res = solve(prices, tasks, pv=pv, export_rate=config.EXPORT_RATE_GBP)
    assert res["status"] == "optimal"
    for t in range(T):  # never import and export simultaneously
        assert min(res["import_kwh"][t], res["export_kwh"][t]) == pytest.approx(0.0, abs=1e-6)
    check = evaluate_schedule(res["starts"], tasks, prices,
                              export_rate=config.EXPORT_RATE_GBP, pv=pv)
    assert check["violations"] == []
    assert res["objective"] == pytest.approx(check["net_cost"], abs=1e-6)


def test_T3_infeasible_deadline_reported():
    # EV needs 12 slots but must finish by slot 8 -> provably infeasible (S4).
    tasks = standard_tasks(ev_latest_finish=8)
    res = solve(agile_like_prices(), tasks)
    assert res["status"] == "infeasible"
    assert res["starts"] is None


def test_T4_pv_pulls_load_into_generation_window():
    prices = [0.30] * T  # flat: price-only problem is fully tied
    pv = [0.0] * T
    for t in range(20, 32):  # 10:00-16:00
        pv[t] = 1.5
    wm = [ApplianceTask("washing_machine", 2.0, 4)]
    res = solve(prices, wm, pv=pv, export_rate=config.EXPORT_RATE_GBP)
    assert res["status"] == "optimal"
    s = res["starts"]["washing_machine"]
    assert 20 <= s and s + 4 <= 32, "WM must sit inside the PV window"
    blind = evaluate_schedule({"washing_machine": 0}, wm, prices,
                              export_rate=config.EXPORT_RATE_GBP, pv=pv)
    aware = evaluate_schedule(res["starts"], wm, prices,
                              export_rate=config.EXPORT_RATE_GBP, pv=pv)
    assert aware["net_cost"] < blind["net_cost"] - 1e-6
    assert aware["self_consumption_ratio"] > blind["self_consumption_ratio"]
