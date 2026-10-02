"""
Phase A, step A4 -- Violation severity in Experiment 2 (constraint conflicts).

Why
---
Table 5 reports how often each model succeeds, but not how badly it fails.
Reviewer 3 asked for violation severity (minutes past a deadline, kW over the
cap) rather than frequency alone. Severity also matters for the deployment
argument: a schedule 30 minutes late and one 10 hours late are both
"failures" in Table 5.

What this script does
---------------------
Uses the A1 deadline rule (deadline = the stated time). For every Exp 2 run it
classifies the outcome and measures each violated constraint:

  correct          success (S1-S3, S5, S6) or infeasibility reported (S4)
  invalid commit   at least one committed appliance violates a stated
                   constraint, or any schedule committed for an infeasible
                   S4 request (fabrication)
  incomplete       some appliances committed, none violating, others missing
  no action        nothing committed and (for S4) nothing reported

  deadline overrun   minutes the EV finishes after the stated deadline
  cap excess         kW above the stated power cap in the worst slot
  quiet-hours breach minutes a noisy appliance starts before the allowed time

Cross-checks the classification against the A1 rescoring before writing.

No LLM calls. Run from the repo root (after A1):
    python -m revision.a4_exp2_violation_severity

Outputs (revision/outputs/):
    a4_severity_by_model.csv   per model and prompt
    a4_invalid_runs.csv        every invalid commit with its severity
    a4_table5_columns.tex      per-model columns to add to Table 5
    a4_summary.txt             console summary
"""
from __future__ import annotations

import collections
import copy
import csv
import json
import statistics as st

from experiments import config
from revision.a1_rescore_departure_buffer import (
    load_scenarios, detect_buffered, rescore, correct, family, slot_time,
    RUNS, MODELS, PROMPTS)

ROOT = config.ROOT
OUT = ROOT / "revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
APP = config.APPLIANCES
NOISY = ("washing_machine", "dishwasher")
T = 48


def starts_of(row) -> dict:
    return {a: v["slot"] for a, v in (row["committed"] or {}).items()
            if v and v.get("slot") is not None}


def measure(starts: dict, cons: dict) -> dict:
    """Severity of each violated constraint for one committed schedule."""
    sev = {"deadline_overrun_min": 0, "cap_excess_kw": 0.0, "quiet_breach_min": 0}
    lf = cons.get("ev_latest_finish")
    if lf is not None and "ev_charger" in starts:
        over = starts["ev_charger"] + APP["ev_charger"]["slots"] - lf
        sev["deadline_overrun_min"] = max(0, over) * 30
    cap = cons.get("power_cap_kw")
    if cap is not None:
        load = [0.0] * T
        for a, s in starts.items():
            for t in range(s, min(T, s + APP[a]["slots"])):
                load[t] += APP[a]["power_kw"]
        sev["cap_excess_kw"] = round(max(0.0, max(load) - cap), 2)
    early = cons.get("noisy_earliest_start")
    if early is not None:
        sev["quiet_breach_min"] = max([max(0, early - starts[a]) * 30
                                       for a in NOISY if a in starts] or [0])
    return sev


def main():
    scen = load_scenarios()
    buffered = detect_buffered(scen)
    rows = [json.loads(l) for l in open(RUNS, encoding="utf-8")]
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("A4 -- Exp 2 violation severity (deadline = stated time, as in A1)")
    say("")

    recs, mismatch = [], 0
    for r in rows:
        sid = r["scenario"]
        cons = copy.deepcopy(scen[sid]["constraints"])
        if sid in buffered:
            cons["ev_latest_finish"] = buffered[sid]
        sc = rescore(r, scen, cons)                     # A1 scoring
        st_ = starts_of(r)
        sev = measure(st_, cons)
        violated = any(v > 0 for v in sev.values())
        if correct(sid, sc):
            outcome = "correct"
        elif family(sid) == "S4":
            outcome = "invalid commit" if st_ else "no action"
        elif violated:
            outcome = "invalid commit"
        elif st_:
            outcome = "incomplete"
        else:
            outcome = "no action"
        # cross-check: outside S4, a violation here <=> A1 scorer reports one
        if family(sid) != "S4" and violated != bool(sc.get("constraint_violations")):
            mismatch += 1
        recs.append({"run_id": r["run_id"], "model": r["model"],
                     "prompt": r["prompt_version"], "scenario": sid,
                     "outcome": outcome, "committed": len(st_), **sev})
    say(f"Cross-check -- severity measures agree with A1 scorer violations on "
        f"{len(rows) - mismatch}/{len(rows)} runs")
    if mismatch:
        raise SystemExit("Severity measures disagree with the scorer; stopping.")
    say("")

    # ------------------------------------------------ outcomes by model
    say("Outcome counts per model (both prompts, 78 runs each):")
    kinds = ["correct", "invalid commit", "incomplete", "no action"]
    say(f"  {'model':10s}" + "".join(f"{k:>16s}" for k in kinds))
    for m in MODELS:
        c = collections.Counter(x["outcome"] for x in recs if x["model"] == m)
        say(f"  {m:10s}" + "".join(f"{c[k]:16d}" for k in kinds))
    say("")

    # ------------------------------------------------ severity
    def summarise(sub):
        # Severity is measured on FEASIBLE scenarios only. In S4 any commit is
        # wrong (fabrication), so its "excess" says nothing about how badly a
        # feasible constraint was missed; S4 fabrications are counted separately.
        inv_all = [x for x in sub if x["outcome"] == "invalid commit"]
        inv = [x for x in inv_all if family(x["scenario"]) != "S4"]
        committed = [x for x in sub if x["committed"] > 0]
        late = [x["deadline_overrun_min"] for x in inv if x["deadline_overrun_min"] > 0]
        cap = [x["cap_excess_kw"] for x in inv if x["cap_excess_kw"] > 0]
        quiet = [x["quiet_breach_min"] for x in inv if x["quiet_breach_min"] > 0]
        fab = len(inv_all) - len(inv)
        return {"runs": len(sub), "committing_runs": len(committed),
                "invalid_commits": len(inv_all), "invalid_feasible_scenarios": len(inv),
                "s4_fabricated": fab,
                "deadline_violations": len(late),
                "max_overrun_h": round(max(late) / 60, 1) if late else 0,
                "median_overrun_h": round(st.median(late) / 60, 1) if late else 0,
                "cap_violations": len(cap), "max_cap_excess_kw": max(cap) if cap else 0,
                "quiet_violations": len(quiet),
                "max_quiet_breach_min": max(quiet) if quiet else 0}

    out_rows = []
    say("Invalid commits per model (both prompts). Severity columns cover feasible")
    say("scenarios (S1-S3, S5, S6); S4 fabrications are counted separately.")
    say(f"  {'model':10s} {'invalid/committing':>19s} {'S4 fab.':>8s} {'late runs':>10s} "
        f"{'max late h':>11s} {'median late h':>14s} {'cap runs':>9s} {'max kW over':>12s} "
        f"{'quiet runs':>11s}")
    for m in MODELS:
        s = summarise([x for x in recs if x["model"] == m])
        out_rows.append({"model": m, "prompt": "both", **s})
        say(f"  {m:10s} {str(s['invalid_commits']) + '/' + str(s['committing_runs']):>19s} "
            f"{s['s4_fabricated']:8d} {s['deadline_violations']:10d} "
            f"{s['max_overrun_h']:11.1f} {s['median_overrun_h']:14.1f} "
            f"{s['cap_violations']:9d} {s['max_cap_excess_kw']:12.1f} {s['quiet_violations']:11d}")
        for p in PROMPTS:
            out_rows.append({"model": m, "prompt": p,
                             **summarise([x for x in recs if x["model"] == m and x["prompt"] == p])})
    tot = summarise(recs)
    say(f"  {'all':10s} {str(tot['invalid_commits']) + '/' + str(tot['committing_runs']):>19s}")
    say("")

    say("Where the violations occur (feasible scenarios; model x runs, severity):")
    by = collections.defaultdict(list)
    for x in recs:
        if x["outcome"] != "invalid commit" or family(x["scenario"]) == "S4":
            continue
        if x["deadline_overrun_min"] > 0:
            by[x["scenario"]].append((x["model"], f"{x['deadline_overrun_min'] / 60:.1f} h late"))
        if x["cap_excess_kw"] > 0:
            by[x["scenario"]].append((x["model"], f"{x['cap_excess_kw']:.1f} kW over cap"))
        if x["quiet_breach_min"] > 0:
            by[x["scenario"]].append((x["model"], f"{x['quiet_breach_min']} min before quiet hours end"))
    for sid in sorted(by):
        c = collections.Counter(by[sid])
        say(f"  {sid:4s} " + ", ".join(f"{m} x{n} ({d})" for (m, d), n in sorted(c.items())))
    say("")

    # ------------------------------------------------ write
    with open(OUT / "a4_severity_by_model.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    inv = [x for x in recs if x["outcome"] == "invalid commit"]
    with open(OUT / "a4_invalid_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(inv)
    tex = ["% Columns to add to Table 5 (both prompts pooled):",
           "% model | invalid commits / committing runs | of which S4 fabrications |",
           "% worst deadline overrun (h) | worst cap excess (kW), feasible scenarios only"]
    for r_ in [o for o in out_rows if o["prompt"] == "both"]:
        late = f"{r_['max_overrun_h']:.1f}" if r_["deadline_violations"] else "--"
        cap = f"{r_['max_cap_excess_kw']:.1f}" if r_["cap_violations"] else "--"
        tex.append(f"{r_['model']} & {r_['invalid_commits']}/{r_['committing_runs']} & "
                   f"{r_['s4_fabricated']} & {late} & {cap} " + r"\\")
    (OUT / "a4_table5_columns.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (OUT / "a4_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote revision/outputs/a4_severity_by_model.csv, a4_invalid_runs.csv, "
          "a4_table5_columns.tex, a4_summary.txt")


if __name__ == "__main__":
    main()
