"""
MILP ground-truth optimizer (paper Appendix A).

Solves: min J = sum_t [ C_t * m_t - F * e_t ]
subject to the energy balance m_t - e_t = L_t + B_t - G_t, one start per
task, horizon/deadline windows, optional household power cap, and --- only
on slots where C_t < F (possible under Agile negative prices) --- explicit
complementarity binaries so the relaxation cannot profit from fictitious
simultaneous import and export.

The same `solve()` covers every ground truth in the paper:
  * daily price-only MILP ........ pv=None, cap=None  (decomposes; test 1)
  * daily extended MILP .......... pv=..., export_rate=...
  * Exp 2 conflicts .............. latest_finish / power_cap_kw set
  * Exp 4b weekly MILP ........... T=336 prices, one ApplianceTask per
                                   required cycle, distinct_day_groups to
                                   forbid two cycles of a type on one day
`evaluate_schedule()` prices ANY committed schedule (agent, rule baseline,
or MILP itself) under identical accounting -- the scorer's core.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pulp

from experiments import config


@dataclass
class ApplianceTask:
    """One schedulable run (a weekly plan uses one task per cycle)."""
    name: str
    power_kw: float
    slots: int
    earliest_start: int = 0
    latest_finish: Optional[int] = None   # require start + slots <= latest_finish


def _feasible_starts(task: ApplianceTask, T: int) -> List[int]:
    lf = task.latest_finish if task.latest_finish is not None else T
    hi = min(T, lf) - task.slots
    return list(range(max(0, task.earliest_start), hi + 1)) if hi >= task.earliest_start else []


def solve(
    prices: Sequence[float],
    tasks: Sequence[ApplianceTask],
    *,
    export_rate: float = 0.0,
    pv: Optional[Sequence[float]] = None,          # kWh per slot
    base: Optional[Sequence[float] | float] = None, # kWh per slot
    power_cap_kw: Optional[float] = None,
    slot_hours: float = config.SLOT_HOURS,
    distinct_day_groups: Optional[List[List[int]]] = None,  # task indices
    slots_per_day: int = config.SLOTS_PER_DAY,
) -> Dict:
    """Return {'status', 'starts', 'objective', 'import_kwh', 'export_kwh'}.

    status is 'optimal' or 'infeasible'. 'infeasible' is returned both when
    a task has no feasible start window (detected pre-solve; this is the
    Exp 2 / S4 ground truth) and when CBC proves joint infeasibility.
    """
    T = len(prices)
    G = [0.0] * T if pv is None else [float(g) for g in pv]
    if base is None:
        B = [config.BASE_LOAD_KWH_PER_SLOT] * T
    elif isinstance(base, (int, float)):
        B = [float(base)] * T
    else:
        B = [float(b) for b in base]
    assert len(G) == T and len(B) == T

    starts_by_task = [_feasible_starts(a, T) for a in tasks]
    if any(not s for s in starts_by_task):
        bad = [tasks[i].name for i, s in enumerate(starts_by_task) if not s]
        return {"status": "infeasible", "reason": f"no feasible start window for {bad}",
                "starts": None, "objective": None}

    prob = pulp.LpProblem("hems", pulp.LpMinimize)

    x = {(i, t): pulp.LpVariable(f"x_{i}_{t}", cat="Binary")
         for i, feas in enumerate(starts_by_task) for t in feas}
    for i, feas in enumerate(starts_by_task):
        prob += pulp.lpSum(x[i, t] for t in feas) == 1, f"one_start_{i}"

    def running(i: int, t: int):
        a = tasks[i]
        return pulp.lpSum(x[i, k] for k in range(max(0, t - a.slots + 1), t + 1)
                          if (i, k) in x)

    m = {t: pulp.LpVariable(f"m_{t}", lowBound=0) for t in range(T)}
    e = {t: pulp.LpVariable(f"e_{t}", lowBound=0, upBound=G[t]) for t in range(T)}

    max_load_kw = power_cap_kw if power_cap_kw is not None else sum(a.power_kw for a in tasks)

    for t in range(T):
        L_t = pulp.lpSum(tasks[i].power_kw * slot_hours * running(i, t)
                         for i in range(len(tasks)))
        prob += m[t] - e[t] == L_t + B[t] - G[t], f"balance_{t}"

        # Complementarity binaries only where the relaxation could cheat
        # (C_t < F with exportable PV): m<=Mm*z, e<=G*(1-z).
        if prices[t] < export_rate and G[t] > 0:
            z = pulp.LpVariable(f"z_{t}", cat="Binary")
            Mm = max_load_kw * slot_hours + B[t]          # m <= L+B always
            prob += m[t] <= Mm * z, f"compl_m_{t}"
            prob += e[t] <= G[t] * (1 - z), f"compl_e_{t}"

        if power_cap_kw is not None:
            prob += (pulp.lpSum(tasks[i].power_kw * running(i, t)
                                for i in range(len(tasks))) <= power_cap_kw,
                     f"cap_{t}")

    if distinct_day_groups:
        n_days = T // slots_per_day
        for g_idx, group in enumerate(distinct_day_groups):
            for d in range(n_days):
                lo, hi = d * slots_per_day, (d + 1) * slots_per_day
                prob += (pulp.lpSum(x[i, t] for i in group
                                    for t in range(lo, hi) if (i, t) in x) <= 1,
                         f"distinct_{g_idx}_{d}")

    prob += pulp.lpSum(prices[t] * m[t] - export_rate * e[t] for t in range(T))

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return {"status": pulp.LpStatus[prob.status].lower(),
                "starts": None, "objective": None}

    starts = {}
    for i, a in enumerate(tasks):
        starts[a.name] = next(t for t in starts_by_task[i] if x[i, t].value() > 0.5)
    return {
        "status": "optimal",
        "starts": starts,
        "objective": pulp.value(prob.objective),
        "import_kwh": [m[t].value() for t in range(T)],
        "export_kwh": [e[t].value() for t in range(T)],
    }


def evaluate_schedule(
    starts: Dict[str, int],
    tasks: Sequence[ApplianceTask],
    prices: Sequence[float],
    *,
    export_rate: float = 0.0,
    pv: Optional[Sequence[float]] = None,
    base: Optional[Sequence[float] | float] = None,
    slot_hours: float = config.SLOT_HOURS,
) -> Dict:
    """Price a committed schedule under Eq. (1) accounting.

    Used identically for agent commitments, rule baselines, and MILP
    output, so every policy is scored by the same function. Also reports
    constraint violations (late finish / horizon overrun) without hiding
    them in the cost.
    """
    T = len(prices)
    G = [0.0] * T if pv is None else list(pv)
    if base is None:
        B = [config.BASE_LOAD_KWH_PER_SLOT] * T
    elif isinstance(base, (int, float)):
        B = [float(base)] * T
    else:
        B = list(base)

    load = [0.0] * T
    violations = []
    by_task = {a.name: a for a in tasks}
    for name, s in starts.items():
        a = by_task[name]
        if s < a.earliest_start or s + a.slots > T:
            violations.append(f"{name}: start {s} outside horizon")
        if a.latest_finish is not None and s + a.slots > a.latest_finish:
            violations.append(f"{name}: finishes after deadline slot {a.latest_finish}")
        for t in range(s, min(s + a.slots, T)):
            load[t] += a.power_kw * slot_hours

    J, imp, exp_ = 0.0, [0.0] * T, [0.0] * T
    self_consumed = 0.0
    for t in range(T):
        net = load[t] + B[t] - G[t]
        imp[t] = max(net, 0.0)
        exp_[t] = max(-net, 0.0)
        self_consumed += min(load[t] + B[t], G[t])
        J += prices[t] * imp[t] - export_rate * exp_[t]

    total_g = sum(G)
    return {
        "net_cost": J,
        "import_kwh": imp,
        "export_kwh": exp_,
        "self_consumption_ratio": (self_consumed / total_g) if total_g > 0 else None,
        "violations": violations,
    }


def cheapest_window(prices: Sequence[float], slots: int,
                    earliest: int = 0, latest_finish: Optional[int] = None) -> Dict:
    """Independent per-appliance minimum-cost window (brute force).

    This is the anchor paper's decomposed ground truth and the Exp 1
    equivalence reference; also reused by the greedy rule baseline.
    """
    T = len(prices)
    lf = latest_finish if latest_finish is not None else T
    best_t, best_sum = None, None
    for t in range(earliest, min(T, lf) - slots + 1):
        s = sum(prices[t:t + slots])
        if best_sum is None or s < best_sum - 1e-12:
            best_t, best_sum = t, s
    return {"start": best_t, "window_sum": best_sum}
