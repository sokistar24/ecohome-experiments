"""
Phase A, step A6 -- Weekly joint planning (Exp 4b): what the logs support.

Why
---
Section 5.5.2 states that GPT-4o-mini "completes all ten repetitions and
commits every requested cycle", that the plans are "feasible and near-optimal",
and that the remaining shortfall comes from cycle-to-day allocation. Reviewers 1
and 3 asked for numbers. The weekly runs were never scored (analyze.py has no
Exp 4b scorer), and RUN.committed keeps only the LAST commit per appliance name
(schedule_appliance overwrites), so each log row holds 3 commitments, not 10.

What this script does
---------------------
1. Rebuilds every weekly plan from the schedule_appliance tool calls (last
   call per appliance and date wins), and checks the rebuild against the
   three commitments that RUN.committed did keep.
2. Checks each plan against the request: 2 washing-machine and 3 dishwasher
   cycles on distinct days, and an EV charge each weekday finishing by the
   stated 08:00. Counts missing and unrequested cycles.
3. Scores the realized weekly cost (actual PV) of exactly what was committed
   and compares it with the weekly MILP. Two MILP references are shown:
   the one in baselines.json (EV by 07:30, as coded) and one re-solved with the
   08:00 deadline the request actually states.
4. For runs that committed every requested cycle, drops the unrequested ones
   (keeping the cheapest valid subset, i.e. the agent's best case) and splits
   the remaining gap into
       day allocation  = MILP with the agent's days fixed - weekly MILP
       within-day timing = agent's subset cost - MILP with the agent's days fixed
   which tests the paper's claim that the shortfall is mainly day allocation.

No LLM calls. Run from the repo root:
    python -m revision.a6_weekly_planning_check

Outputs (revision/outputs/):
    a6_weekly_runs.csv   one row per run
    a6_summary.txt       console summary
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
from datetime import date

from experiments import archive, config
from experiments.optimizer import ApplianceTask, evaluate_schedule, solve

ROOT = config.ROOT
RUNS = ROOT / "data" / "runs" / "exp4b.jsonl"
BASELINES = ROOT / "data" / "results" / "baselines.json"
OUT = ROOT / "revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

T = config.SLOTS_PER_DAY
APP = config.APPLIANCES
F = config.EXPORT_RATE_GBP
STATED_EV_FINISH = 16            # "ready by 08:00" in the exp4b request
CODED_EV_FINISH = 15             # EV_LATEST_FINISH (07:30) used by weekly_oracle
NEED = {"washing_machine": 2, "dishwasher": 3}


def slot_of(args) -> int:
    if args.get("start_slot") is not None:
        return int(args["start_slot"])
    hh, mm = args["start_time"].strip().split(":")
    return int(hh) * 2 + (1 if int(mm) >= 30 else 0)


def weekly_tasks(ev_finish: int):
    """Same structure as compute_baselines.weekly_task_set, with the EV
    deadline as a parameter."""
    tasks, wm, dw = [], [], []
    for j in range(2):
        tasks.append(ApplianceTask(f"wm_cycle{j}", APP["washing_machine"]["power_kw"],
                                   APP["washing_machine"]["slots"])); wm.append(len(tasks) - 1)
    for j in range(3):
        tasks.append(ApplianceTask(f"dw_cycle{j}", APP["dishwasher"]["power_kw"],
                                   APP["dishwasher"]["slots"])); dw.append(len(tasks) - 1)
    for d in range(5):
        base = d * T
        tasks.append(ApplianceTask(f"ev_day{d}", APP["ev_charger"]["power_kw"],
                                   APP["ev_charger"]["slots"],
                                   earliest_start=max(0, base - T // 2),
                                   latest_finish=base + ev_finish))
    return tasks, [wm, dw]


def weekly_milp(week, ev_finish):
    prices = archive.week_prices(week)
    pv_fc, pv_act = archive.week_pv(week, "fc"), archive.week_pv(week, "actual")
    tasks, groups = weekly_tasks(ev_finish)
    res = solve(prices, tasks, pv=pv_fc, export_rate=F, distinct_day_groups=groups)
    assert res["status"] == "optimal", res["status"]
    return evaluate_schedule(res["starts"], tasks, prices, export_rate=F, pv=pv_act)["net_cost"]


def main():
    sel = archive.load_selection()
    weeks = {f"week{i}": w for i, w in enumerate(sel["exp4_weeks"])}
    base = json.loads(BASELINES.read_text(encoding="utf-8"))["weekly_oracle"]
    coded = {w["week_start"]: w["net_cost_realized"] for w in base}
    rows = [json.loads(l) for l in open(RUNS, encoding="utf-8")]
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"A6 -- Exp 4b weekly planning ({len(rows)} runs, model: "
        f"{', '.join(sorted({r['model'] for r in rows}))})")
    say("")

    milp = {}
    for wk, days in weeks.items():
        if not any(r["scenario"] == wk for r in rows):
            continue
        c15 = weekly_milp(days, CODED_EV_FINISH)
        assert abs(c15 - coded[days[0]]) < 1e-3, "weekly MILP does not reproduce baselines.json"
        milp[wk] = {"coded_0730": c15, "stated_0800": weekly_milp(days, STATED_EV_FINISH)}
    say("Sanity check 1 -- weekly MILP (EV by 07:30) reproduces baselines.json")
    for wk, v in milp.items():
        say(f"  {wk} ({weeks[wk][0]}): MILP GBP {v['coded_0730']:.2f} with EV by 07:30 as coded, "
            f"GBP {v['stated_0800']:.2f} with EV by 08:00 as requested")
    say("")

    out, rebuilt_ok = [], 0
    for r in rows:
        days = weeks[r["scenario"]]
        idx = {d: i for i, d in enumerate(days)}
        last = {}
        for c in r["tool_calls"]:
            if c["name"] == "schedule_appliance":
                a = c["args"]
                last[(a["appliance"], a["date"])] = slot_of(a)
        # rebuild check: the final commit per appliance must equal RUN.committed
        final_by_app = {}
        for c in r["tool_calls"]:
            if c["name"] == "schedule_appliance":
                final_by_app[c["args"]["appliance"]] = (c["args"]["date"], slot_of(c["args"]))
        kept = {a: (v["date"], v["slot"]) for a, v in (r["committed"] or {}).items()}
        rebuilt_ok += final_by_app == kept

        commits = [(a, d, s) for (a, d), s in last.items() if d in idx]
        outside = [(a, d) for (a, d) in last if d not in idx]
        n = collections.Counter(a for a, _, _ in commits)
        ev_days = {idx[d] for a, d, _ in commits if a == "ev_charger"}
        weekdays = {i for i, d in enumerate(days) if date.fromisoformat(d).weekday() < 5}
        missing = {a: max(0, k - n[a]) for a, k in NEED.items()}
        missing["ev_charger"] = len(weekdays - ev_days)
        extra = {a: max(0, n[a] - k) for a, k in NEED.items()}
        extra["ev_charger"] = len(ev_days - weekdays)
        late = [days[idx[d]] for a, d, s in commits
                if a == "ev_charger" and idx[d] in weekdays
                and s + APP["ev_charger"]["slots"] > STATED_EV_FINISH]

        tasks, starts = [], {}
        for k, (a, d, s) in enumerate(commits):
            name = f"{a}_{k}"
            tasks.append(ApplianceTask(name, APP[a]["power_kw"], APP[a]["slots"]))
            starts[name] = idx[d] * T + s
        prices = archive.week_prices(days)
        pv_act = archive.week_pv(days, "actual")
        cost = evaluate_schedule(starts, tasks, prices, export_rate=F, pv=pv_act)["net_cost"]

        # ---- requested subset and day-vs-timing decomposition (step 4)
        decomp = {"subset_cost_gbp": "", "day_alloc_gbp": "", "timing_gbp": ""}
        if not any(missing.values()):
            best = None
            pools = {a: [(d, s_) for a2, d, s_ in commits if a2 == a] for a in NEED}
            evs = [(d, s_) for a2, d, s_ in commits if a2 == "ev_charger" and idx[d] in weekdays]
            for wm in itertools.combinations(pools["washing_machine"], 2):
                for dw in itertools.combinations(pools["dishwasher"], 3):
                    if len({d for d, _ in wm}) < 2 or len({d for d, _ in dw}) < 3:
                        continue
                    chosen = ([("washing_machine", d, s_) for d, s_ in wm] +
                              [("dishwasher", d, s_) for d, s_ in dw] +
                              [("ev_charger", d, s_) for d, s_ in evs])
                    tk, stt = [], {}
                    for k, (a, d, s_) in enumerate(chosen):
                        tk.append(ApplianceTask(f"{a}_{k}", APP[a]["power_kw"], APP[a]["slots"]))
                        stt[f"{a}_{k}"] = idx[d] * T + s_
                    c_ = evaluate_schedule(stt, tk, prices, export_rate=F, pv=pv_act)["net_cost"]
                    if best is None or c_ < best[0]:
                        best = (c_, chosen)
            if best is not None:
                # MILP with the agent's day assignments fixed, slots free within each day
                fixed = []
                for k, (a, d, _) in enumerate(best[1]):
                    i = idx[d]
                    if a == "ev_charger":
                        fixed.append(ApplianceTask(f"ev_{k}", APP[a]["power_kw"], APP[a]["slots"],
                                                   earliest_start=max(0, i * T - T // 2),
                                                   latest_finish=i * T + STATED_EV_FINISH))
                    else:
                        fixed.append(ApplianceTask(f"{a}_{k}", APP[a]["power_kw"], APP[a]["slots"],
                                                   earliest_start=i * T,
                                                   latest_finish=(i + 1) * T))
                pv_fc = archive.week_pv(days, "fc")
                res = solve(prices, fixed, pv=pv_fc, export_rate=F)
                assert res["status"] == "optimal"
                j_fixed = evaluate_schedule(res["starts"], fixed, prices,
                                            export_rate=F, pv=pv_act)["net_cost"]
                ref_ = milp[r["scenario"]]["stated_0800"]
                decomp = {"subset_cost_gbp": round(best[0], 3),
                          "day_alloc_gbp": round(j_fixed - ref_, 3),
                          "timing_gbp": round(best[0] - j_fixed, 3)}

        complete = not any(missing.values())
        exact = complete and not any(extra.values()) and not late
        ref = milp[r["scenario"]]["stated_0800"]
        out.append({"run_id": r["run_id"], "week": r["scenario"],
                    "schedule_calls": sum(c["name"] == "schedule_appliance" for c in r["tool_calls"]),
                    "wm": n["washing_machine"], "dw": n["dishwasher"], "ev_days": len(ev_days),
                    "missing": sum(missing.values()), "unrequested": sum(extra.values()),
                    "ev_late": len(late), "outside_week": len(outside),
                    "as_requested": exact,
                    "cost_gbp": round(cost, 3), "milp_0800_gbp": round(ref, 3),
                    "gap_gbp": round(cost - ref, 3),
                    "gap_pct": round(100 * (cost - ref) / abs(ref), 1),
                    **decomp,
                    "tokens": r["tokens"]["input"] + r["tokens"]["output"],
                    "latency_s": r["latency_s"], "usd": r["est_cost_usd"]})
    say(f"Sanity check 2 -- rebuilt plans agree with RUN.committed's last commit per "
        f"appliance in {rebuilt_ok}/{len(rows)} runs")
    say("")

    say("Per run (WM/DW counts, weekday EV charges; MILP = EV by 08:00 as requested):")
    say(f"  {'week':6s} {'WM':>3s} {'DW':>3s} {'EV':>3s} {'missing':>8s} {'unrequested':>12s} "
        f"{'late EV':>8s} {'as asked':>9s} {'cost £':>7s} {'MILP £':>7s} {'gap £':>6s} "
        f"{'gap %':>6s} {'tokens':>7s} {'s':>5s}")
    for o in out:
        say(f"  {o['week']:6s} {o['wm']:3d} {o['dw']:3d} {o['ev_days']:3d} {o['missing']:8d} "
            f"{o['unrequested']:12d} {o['ev_late']:8d} {str(o['as_requested']):>9s} "
            f"{o['cost_gbp']:7.2f} {o['milp_0800_gbp']:7.2f} {o['gap_gbp']:6.2f} "
            f"{o['gap_pct']:6.1f} {o['tokens']:7d} {o['latency_s']:5.0f}")
    say("")
    ok = [o for o in out if o["as_requested"]]
    say(f"Runs that committed exactly what was requested: {len(ok)}/{len(out)}")
    say(f"Runs with missing cycles: {sum(o['missing'] > 0 for o in out)}; with unrequested "
        f"cycles (e.g. weekend EV charges): {sum(o['unrequested'] > 0 for o in out)}")
    if ok:
        g = sorted(o["gap_pct"] for o in ok)
        say(f"Gap to the weekly MILP among those runs: {g[0]:.1f}% to {g[-1]:.1f}% "
            f"(GBP {min(o['gap_gbp'] for o in ok):.2f} to {max(o['gap_gbp'] for o in ok):.2f})")
    say("")
    dec = [o for o in out if o["subset_cost_gbp"] != ""]
    if dec:
        say("Requested cycles only (unrequested ones dropped, agent's best case), gap to the")
        say("weekly MILP split into day allocation and within-day timing, GBP/week:")
        say(f"  {'week':6s} {'subset £':>9s} {'gap £':>7s} {'gap %':>6s} {'day alloc £':>12s} {'timing £':>9s}")
        for o in dec:
            gap = o["subset_cost_gbp"] - o["milp_0800_gbp"]
            say(f"  {o['week']:6s} {o['subset_cost_gbp']:9.2f} {gap:7.2f} "
                f"{100 * gap / o['milp_0800_gbp']:6.1f} "
                f"{o['day_alloc_gbp']:12.2f} {o['timing_gbp']:9.2f}")
        tot_d = sum(o["day_alloc_gbp"] for o in dec)
        tot_t = sum(o["timing_gbp"] for o in dec)
        say(f"  share of the gap from day allocation: {100 * tot_d / (tot_d + tot_t):.0f}%")
        say("")
    say("Note: the weekly request asks for the EV 'ready by 08:00' but weekly_oracle in")
    say("compute_baselines.py uses EV_LATEST_FINISH (07:30). Gaps above use the stated 08:00.")

    with open(OUT / "a6_weekly_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    (OUT / "a6_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote revision/outputs/a6_weekly_runs.csv, a6_summary.txt")


if __name__ == "__main__":
    main()
