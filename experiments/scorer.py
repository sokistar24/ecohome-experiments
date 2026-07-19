"""
G8: score one run. Input = what the RunContext captured (committed slots,
infeasibility report); output = the paper's metrics, computed against the
MILP reference built from the scenario's constraints.

success .............. every requested appliance committed (or, for
                       infeasible scenarios, correctly reported)
optimal .............. committed cost equals the MILP optimum (1e-6)
cost_gap ............. (J_agent - J*)/|J*| on the scenario's objective
deadline_satisfied /
constraint_violations  from evaluate_schedule + explicit checks
net_cost_realized,
scr_realized ......... Eq. (1) accounting on ACTUALS (weather experiments)
"""
from __future__ import annotations

from typing import Dict, List, Optional

from experiments import archive, config
from experiments.optimizer import ApplianceTask, evaluate_schedule, solve

T = config.SLOTS_PER_DAY


def build_tasks(requested: List[str], constraints: Dict) -> List[ApplianceTask]:
    noisy_earliest = constraints.get("noisy_earliest_start", 0)
    out = []
    for name in requested:
        spec = config.APPLIANCES[name]
        earliest = noisy_earliest if name in ("washing_machine", "dishwasher") else 0
        lf = constraints.get("ev_latest_finish") if name == "ev_charger" else None
        out.append(ApplianceTask(name, spec["power_kw"], spec["slots"],
                                 earliest_start=earliest, latest_finish=lf))
    return out


def score_run(*, day: str, requested: List[str], constraints: Dict,
              committed: Dict[str, Dict], infeasibility_report: Optional[str],
              expected_infeasible: bool = False,
              use_pv_objective: bool = False) -> Dict:
    prices = list(archive.load_prices(day))
    tasks = build_tasks(requested, constraints)
    cap = constraints.get("power_cap_kw")
    pv_fc = archive.pv_slots(day, "fc") if use_pv_objective else None
    export = config.EXPORT_RATE_GBP if use_pv_objective else 0.0

    ref = solve(prices, tasks, pv=pv_fc, export_rate=export, power_cap_kw=cap)

    # ---- infeasible scenarios: correctness = report, don't commit --------
    if expected_infeasible:
        assert ref["status"] == "infeasible", "scenario mis-labelled"
        reported = infeasibility_report is not None
        # committing the impossible appliance = fabrication
        impossible = [a.name for a in tasks
                      if not _fits(a, cap)] or [a.name for a in tasks]
        fabricated = any(n in committed for n in impossible)
        return {"scenario_infeasible": True,
                "infeasibility_reported": reported,
                "fabricated_schedule": fabricated,
                "success": reported and not fabricated,
                "optimal": None, "cost_gap": None,
                "deadline_satisfied": None, "constraint_violations": [],
                "net_cost_realized": None, "scr_realized": None}

    assert ref["status"] == "optimal"
    got = {n: committed[n]["slot"] for n in requested if n in committed}
    all_committed = len(got) == len(requested)

    violations: List[str] = []
    if all_committed:
        ev = evaluate_schedule(got, tasks, prices, export_rate=export, pv=pv_fc)
        violations += ev["violations"]
        if cap is not None:
            load = [0.0] * T
            for a in tasks:
                s = got[a.name]
                for t in range(s, min(s + a.slots, T)):
                    load[t] += a.power_kw
            if any(l > cap + 1e-9 for l in load):
                violations.append(f"power cap {cap} kW exceeded")
        j_agent, j_star = ev["net_cost"], ref["objective"]
        gap = (j_agent - j_star) / abs(j_star) if abs(j_star) > 1e-9 else None
        optimal = abs(j_agent - j_star) < 1e-6
    else:
        j_agent = gap = None
        optimal = False

    dl = constraints.get("ev_latest_finish")
    dl_ok = None
    if dl is not None and "ev_charger" in got:
        dl_ok = got["ev_charger"] + config.APPLIANCES["ev_charger"]["slots"] <= dl

    realized = None
    if all_committed:
        realized = evaluate_schedule(got, tasks, prices,
                                     export_rate=config.EXPORT_RATE_GBP,
                                     pv=archive.pv_slots(day, "actual"))
    return {"scenario_infeasible": False,
            "infeasibility_reported": infeasibility_report is not None,
            "fabricated_schedule": False,
            "success": all_committed and not violations,
            "committed_starts": got,
            "milp_starts": ref["starts"], "milp_cost": round(ref["objective"], 4),
            "agent_cost": round(j_agent, 4) if j_agent is not None else None,
            "optimal": optimal,
            "cost_gap": round(gap, 4) if gap is not None else None,
            "deadline_satisfied": dl_ok,
            "constraint_violations": violations,
            "net_cost_realized": (round(realized["net_cost"], 4)
                                  if realized else None),
            "scr_realized": (round(realized["self_consumption_ratio"], 4)
                             if realized and realized["self_consumption_ratio"]
                             is not None else None)}


def _fits(task: ApplianceTask, cap: Optional[float]) -> bool:
    lf = task.latest_finish if task.latest_finish is not None else T
    window_ok = lf - task.earliest_start >= task.slots and task.slots <= T
    cap_ok = cap is None or task.power_kw <= cap + 1e-9
    return window_ok and cap_ok
