"""
Phase C, step 1 -- how strongly the appliances are coupled on the evaluation days.

Quoted in the Methods (Section "Net-cost MILP"):
  * Exp 1 days (price-only, no cap): scheduling each appliance on its own
    cheapest window gives exactly the joint MILP optimum, i.e. the problem
    decomposes and Exp 1 tests task completion, not coordination.
  * Exp 3 days (forecast PV, EV by 18:00): scheduling each appliance on its
    own against the full PV forecast differs from the joint optimum, and
    adding a 9 kW cap makes those independent plans infeasible on most days.

No LLM calls. Run from the repo root:
    python -m revision.c1_coupling_check
Output: revision/outputs/c1_coupling.txt
"""
from __future__ import annotations

import statistics as st

from experiments import archive, config
from experiments.compute_baselines import standard_tasks, EV_LATEST_FINISH, EV_WFH_LATEST_FINISH
from experiments.optimizer import cheapest_window, evaluate_schedule, solve

F = config.EXPORT_RATE_GBP
OUT = config.ROOT / "revision" / "outputs" / "c1_coupling.txt"


def main():
    sel = archive.load_selection()
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    gaps = []
    days1 = [d for t in ("low", "mid", "high") for d in sel["exp1_terciles"][t]]
    for d in days1:
        p = list(archive.load_prices(d))
        tasks = standard_tasks(EV_LATEST_FINISH)
        joint = solve(p, tasks)["objective"]
        ind = {a.name: cheapest_window(p, a.slots, a.earliest_start, a.latest_finish)["start"]
               for a in tasks}
        gaps.append(evaluate_schedule(ind, tasks, p)["net_cost"] - joint)
    say(f"Exp 1: independent cheapest windows equal the joint MILP on "
        f"{sum(abs(g) < 1e-9 for g in gaps)}/{len(days1)} days (max difference GBP {max(gaps):.6f})")

    days3 = [d for r in ("sunny", "mixed", "overcast") for d in sel["exp3_regimes"][r]]
    g3, cap_broken = [], 0
    for d in days3:
        p = list(archive.load_prices(d))
        pv = archive.pv_slots(d, "fc")
        tasks = standard_tasks(EV_WFH_LATEST_FINISH)
        joint = solve(p, tasks, pv=pv, export_rate=F)["objective"]
        ind = {a.name: solve(p, [a], pv=pv, export_rate=F)["starts"][a.name] for a in tasks}
        g3.append(evaluate_schedule(ind, tasks, p, export_rate=F, pv=pv)["net_cost"] - joint)
        load = [0.0] * config.SLOTS_PER_DAY
        for a in tasks:
            for t in range(ind[a.name], ind[a.name] + a.slots):
                load[t] += a.power_kw
        cap_broken += max(load) > 9.0 + 1e-9
    say(f"Exp 3: optimising each appliance alone against the full PV forecast differs from the "
        f"joint MILP on {sum(g > 1e-4 for g in g3)}/{len(days3)} days "
        f"(mean GBP {st.mean(g3):.2f}/day, max {max(g3):.2f}); under a 9 kW cap those "
        f"independent plans break the cap on {cap_broken}/{len(days3)} days")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
