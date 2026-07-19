"""
A3: compute every zero-token reference policy over the frozen archives.

    python -m experiments.compute_baselines

For each benchmark day, prices five policies under identical accounting
(optimizer.evaluate_schedule), each scored two ways:
  * cost_price_only ....... Eq. (1) with G=0  (the anchor-paper objective)
  * net_cost_realized ..... Eq. (1) with PV from the ACTUALS archive
Policies:
  off_peak_timer   all appliances start 00:00 (EV first; what timer plugs do)
  immediate        all start 18:00, arrival home (clipped to fit the horizon)
  greedy_slot      start at the day's single cheapest slot, ignoring windows
  price_only_milp  MILP oracle on prices alone (decides with G=0)
  oracle           extended MILP deciding on FORECAST PV, scored on actuals

Also solves the Exp 4b weekly MILP for each selected week (validates the
336-slot model on real data). Output: data/results/baselines.json + a
console summary. These numbers ARE the reference rows of paper Tables
IV-VII; nothing here ever needs an LLM call.

The standard household scenario (also imported by the runner later):
WM + DW unconstrained within the day; EV must finish by 07:30 (slot 15),
i.e. one buffer slot before a 08:00 departure -- so the EV must start by
slot 3 (01:30).
"""
from __future__ import annotations

import json
from statistics import mean
from typing import Dict, List

from experiments import archive, config
from experiments.optimizer import ApplianceTask, evaluate_schedule, solve

T = config.SLOTS_PER_DAY
EV_DEPARTURE_SLOT = 16                       # 08:00
EV_LATEST_FINISH = EV_DEPARTURE_SLOT - config.EV_DEADLINE_BUFFER_SLOTS  # 15 = 07:30
EV_WFH_LATEST_FINISH = 36                    # 18:00 (Exp 3 work-from-home variant)

# Scenario variants. "standard" (07:30 EV readiness) is the Exp 1/2/4
# setting. "wfh_ev18" (18:00 readiness) is the Exp 3 setting: with a 07:30
# deadline the EV -- the only load large enough for solar to matter -- can
# never overlap daylight, structurally capping the weather-aware effect at
# the WM+DW margin; the WFH variant is the realistic case where it can.
SCENARIOS = {"standard": EV_LATEST_FINISH, "wfh_ev18": EV_WFH_LATEST_FINISH}


def standard_tasks(ev_latest_finish: int | None = EV_LATEST_FINISH) -> List[ApplianceTask]:
    a = config.APPLIANCES
    return [
        ApplianceTask("washing_machine", a["washing_machine"]["power_kw"],
                      a["washing_machine"]["slots"]),
        ApplianceTask("dishwasher", a["dishwasher"]["power_kw"],
                      a["dishwasher"]["slots"]),
        ApplianceTask("ev_charger", a["ev_charger"]["power_kw"],
                      a["ev_charger"]["slots"],
                      latest_finish=ev_latest_finish),
    ]


# ------------------------------------------------------------- rule policies
def policy_off_peak_timer(tasks) -> Dict[str, int]:
    """Static timer: everything starts at 00:00 (EV first is cosmetic --
    without a power cap they may run concurrently, as real timer plugs do)."""
    return {a.name: 0 for a in tasks}


def policy_immediate(tasks) -> Dict[str, int]:
    """Run on arrival home at 18:00; clipped so each run fits the horizon.
    Note: violates the EV deadline by construction -- that is the point of
    this baseline, and evaluate_schedule reports the violation."""
    s = 36  # 18:00
    return {a.name: min(s, T - a.slots) for a in tasks}


def policy_greedy_slot(prices, tasks) -> Dict[str, int]:
    """Start everything at the day's single cheapest slot, ignoring window
    sums, deadlines, and the cap (the naive 'cheapest hour' heuristic)."""
    t_star = min(range(T), key=lambda t: prices[t])
    return {a.name: min(t_star, T - a.slots) for a in tasks}


# ------------------------------------------------------------- scoring
def score(starts, tasks, prices, pv_actual) -> Dict:
    price_only = evaluate_schedule(starts, tasks, prices)
    realized = evaluate_schedule(starts, tasks, prices,
                                 export_rate=config.EXPORT_RATE_GBP,
                                 pv=pv_actual)
    return {
        "starts": starts,
        "cost_price_only": round(price_only["net_cost"], 4),
        "net_cost_realized": round(realized["net_cost"], 4),
        "scr_realized": (round(realized["self_consumption_ratio"], 4)
                         if realized["self_consumption_ratio"] is not None else None),
        "violations": realized["violations"],
    }


def baselines_for_day(day: str, ev_latest_finish: int = EV_LATEST_FINISH) -> Dict:
    prices = list(archive.load_prices(day))
    pv_fc = archive.pv_slots(day, "fc")
    pv_act = archive.pv_slots(day, "actual")
    tasks = standard_tasks(ev_latest_finish)

    out: Dict[str, Dict] = {}
    out["off_peak_timer"] = score(policy_off_peak_timer(tasks), tasks, prices, pv_act)
    out["immediate"] = score(policy_immediate(tasks), tasks, prices, pv_act)
    out["greedy_slot"] = score(policy_greedy_slot(prices, tasks), tasks, prices, pv_act)

    po = solve(prices, tasks)                                   # decides with G=0
    if po["status"] != "optimal":
        raise RuntimeError(f"{day}: price-only MILP {po['status']}")
    out["price_only_milp"] = score(po["starts"], tasks, prices, pv_act)

    ox = solve(prices, tasks, pv=pv_fc,                         # decides on forecast
               export_rate=config.EXPORT_RATE_GBP)
    if ox["status"] != "optimal":
        raise RuntimeError(f"{day}: oracle MILP {ox['status']}")
    out["oracle"] = score(ox["starts"], tasks, prices, pv_act)  # scored on actuals
    out["oracle"]["objective_on_forecast"] = round(ox["objective"], 4)
    return out


# ------------------------------------------------------------- weekly (4b)
def weekly_task_set(week_days: List[str]):
    """2 WM cycles + 3 DW cycles anywhere in the week (distinct days per
    type) + EV each weekday finishing by 07:30 local of that day."""
    a = config.APPLIANCES
    tasks, wm_idx, dw_idx = [], [], []
    for j in range(2):
        tasks.append(ApplianceTask(f"wm_cycle{j}", a["washing_machine"]["power_kw"],
                                   a["washing_machine"]["slots"]))
        wm_idx.append(len(tasks) - 1)
    for j in range(3):
        tasks.append(ApplianceTask(f"dw_cycle{j}", a["dishwasher"]["power_kw"],
                                   a["dishwasher"]["slots"]))
        dw_idx.append(len(tasks) - 1)
    for d in range(5):  # Mon-Fri departures
        base = d * T
        tasks.append(ApplianceTask(
            f"ev_day{d}", a["ev_charger"]["power_kw"], a["ev_charger"]["slots"],
            earliest_start=max(0, base - T // 2),
            latest_finish=base + EV_LATEST_FINISH))
    return tasks, [wm_idx, dw_idx]


def weekly_oracle(week_days: List[str]) -> Dict:
    prices = archive.week_prices(week_days)
    pv_fc = archive.week_pv(week_days, "fc")
    pv_act = archive.week_pv(week_days, "actual")
    tasks, groups = weekly_task_set(week_days)
    res = solve(prices, tasks, pv=pv_fc, export_rate=config.EXPORT_RATE_GBP,
                distinct_day_groups=groups)
    if res["status"] != "optimal":
        raise RuntimeError(f"week {week_days[0]}: weekly MILP {res['status']}")
    realized = evaluate_schedule(res["starts"], tasks, prices,
                                 export_rate=config.EXPORT_RATE_GBP, pv=pv_act)
    return {"week_start": week_days[0],
            "starts": res["starts"],
            "objective_on_forecast": round(res["objective"], 4),
            "net_cost_realized": round(realized["net_cost"], 4),
            "violations": realized["violations"]}


# ------------------------------------------------------------- main
def main():
    sel = archive.load_selection()
    days = archive.all_benchmark_days(sel)
    print(f"[baselines] {len(days)} benchmark days")

    per_day = {}
    for d in days:
        per_day[d] = {name: baselines_for_day(d, lf)
                      for name, lf in SCENARIOS.items()}
    weeks = [weekly_oracle(w) for w in sel["exp4_weeks"]]

    out = {"config": {"export_rate_gbp": config.EXPORT_RATE_GBP,
                      "pv_kwp": config.PV_KWP,
                      "ev_latest_finish_slot": EV_LATEST_FINISH},
           "per_day": per_day, "weekly_oracle": weeks}
    results_dir = config.DATA / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "baselines.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[baselines] wrote {path}")

    # ---------------- console summary (means over all benchmark days)
    names = ["immediate", "off_peak_timer", "greedy_slot",
             "price_only_milp", "oracle"]
    for scen in SCENARIOS:
        print(f"\n=== scenario: {scen} ===")
        print(f"{'policy':<18}{'price-only £/day':>18}{'realized net £/day':>20}{'mean SCR':>10}")
        for n in names:
            po = mean(per_day[d][scen][n]["cost_price_only"] for d in days)
            rz = mean(per_day[d][scen][n]["net_cost_realized"] for d in days)
            scrs = [per_day[d][scen][n]["scr_realized"] for d in days
                    if per_day[d][scen][n]["scr_realized"] is not None]
            scr = mean(scrs) if scrs else float("nan")
            print(f"{n:<18}{po:>18.3f}{rz:>20.3f}{scr:>10.3f}")
        timer = mean(per_day[d][scen]["off_peak_timer"]["net_cost_realized"] for d in days)
        oracle = mean(per_day[d][scen]["oracle"]["net_cost_realized"] for d in days)
        pom = mean(per_day[d][scen]["price_only_milp"]["net_cost_realized"] for d in days)
        gap = timer - oracle
        if gap > 0.02:
            print(f"savings gap (timer -> oracle): £{gap:.3f}/day; "
                  f"price-only MILP captures {100 * (timer - pom) / gap:.1f}% of it")
        else:
            print(f"(timer -> oracle gap £{gap:.3f}/day -- too small for share arithmetic)")
    for w in weeks:
        print(f"weekly oracle {w['week_start']}: realized £{w['net_cost_realized']:.3f} "
              f"(violations: {len(w['violations'])})")


if __name__ == "__main__":
    main()
