"""
Phase A, step A1 -- Rescore Experiment 2 without the unstated departure buffer.

Why
---
mine_scenarios.py sets the EV deadline for calendar-based scenarios as
(departure slot - EV_DEADLINE_BUFFER_SLOTS), i.e. 30 minutes before the
calendar time. The agent is never told about this buffer (it appears in no
prompt, tool description or context line). S1 does not use it either:
"ready to leave by 15:30" is scored with latest_finish = 15:30. Runs that
finish exactly at the stated departure time were therefore scored as late.

What this script does
---------------------
1. Detects which scenarios apply the buffer (calendar time - 1 slot == deadline).
2. Re-scores every logged Exp 2 run with the ORIGINAL constraints and checks
   that it reproduces the logged scores exactly (pipeline sanity check).
3. Re-scores every run with the S1 convention: deadline = stated time.
4. Writes Table 5 / Fig 3 inputs and a before/after summary.

No LLM calls; only scoring changes. Run from the repo root:
    python -m revision.a1_rescore_departure_buffer

Outputs (revision/outputs/):
    a1_exp2_main_rescored.csv  same schema as data/results/exp2_main.csv
    a1_changed_runs.csv        every run whose outcome flips
    a1_summary.txt             Table 5 and aggregate compliance, before vs after
"""
from __future__ import annotations

import collections
import copy
import csv
import json
import re
from pathlib import Path

from experiments import archive, config
from experiments.optimizer import cheapest_window
from experiments.scorer import score_run

ROOT = config.ROOT
RUNS = ROOT / "data" / "runs" / "exp2.jsonl"
SCEN_DIR = ROOT / "data" / "archive" / "scenarios"
OUT = ROOT / "revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

ALL_THREE = ["washing_machine", "dishwasher", "ev_charger"]
BUFFER = config.EV_DEADLINE_BUFFER_SLOTS
EXPECTED_BUFFERED = {"S3a", "S3b", "S5"}      # from mine_scenarios.py
MODELS = ["gpt", "gemini", "claude", "llama-3.3", "qwen-3"]
PROMPTS = ["v1", "v2-guided"]
FAMILIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
FAMKEY = {"S1": "deadline_conflict", "S2": "power_cap",
          "S3": "irregular_calendar", "S4": "infeasible",
          "S5": "instruction_vs_calendar", "S6": "tool_failure"}
EV_SLOTS = config.APPLIANCES["ev_charger"]["slots"]


def hhmm_to_slot(t: str) -> int:
    hh, mm = t.split(":")
    return int(hh) * 2 + (1 if int(mm) >= 30 else 0)


def slot_time(s: int) -> str:
    return f"{s // 2:02d}:{(s % 2) * 30:02d}"


def load_scenarios() -> dict:
    manifest = json.loads((SCEN_DIR / "manifest.json").read_text())
    return {e["id"]: json.loads((SCEN_DIR / f"{e['id']}.json").read_text())
            for e in manifest}


def detect_buffered(scen: dict) -> dict:
    """scenario id -> stated departure slot, for scenarios whose deadline is
    exactly one buffer before a calendar time given to the agent."""
    out = {}
    for sid, s in scen.items():
        lf = s["constraints"].get("ev_latest_finish")
        if lf is None:
            continue
        for line in s.get("context_extras") or []:
            if not line.lower().startswith("calendar"):
                continue
            for t in re.findall(r"\b(\d{1,2}:\d{2})\b", line):
                if hhmm_to_slot(t) - BUFFER == lf:
                    out[sid] = hhmm_to_slot(t)
    return out


def rescore(row: dict, scen: dict, constraints: dict) -> dict:
    s = scen[row["scenario"]]
    return score_run(day=s["day"], requested=ALL_THREE, constraints=constraints,
                     committed=row["committed"] or {},
                     infeasibility_report=row["infeasibility_report"],
                     expected_infeasible=s["expected"].get("infeasible", False))


def family(sid: str) -> str:
    return sid[:2]


def correct(sid: str, sc: dict) -> bool:
    """Same definition as build_exp2_main_csv.py / Table 5."""
    if family(sid) == "S4":
        return bool(sc.get("infeasibility_reported")) and not bool(sc.get("fabricated_schedule"))
    return bool(sc.get("success"))


def table5(rows, key):
    """(model, family) -> 'baseline/guided' string; plus per-model aggregate."""
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["model"], r["prompt_version"], family(r["scenario"]))].append(r[key])
    cells = {}
    for m in MODELS:
        for f in FAMILIES:
            v = [sum(agg[(m, p, f)]) / len(agg[(m, p, f)]) for p in PROMPTS]
            cells[(m, f)] = f"{v[0]:.2f} / {v[1]:.2f}"
    overall = {m: (sum(r[key] for r in rows if r["model"] == m),
                   sum(1 for r in rows if r["model"] == m)) for m in MODELS}
    return agg, cells, overall


def main():
    scen = load_scenarios()
    buffered = detect_buffered(scen)
    assert set(buffered) == EXPECTED_BUFFERED, (
        f"buffered scenarios detected {sorted(buffered)}, expected "
        f"{sorted(EXPECTED_BUFFERED)} -- check the scenario files")

    rows = [json.loads(l) for l in open(RUNS, encoding="utf-8")]
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say("A1 -- Exp 2 rescoring without the unstated 30-min departure buffer")
    say(f"runs: {len(rows)}")
    say("")
    say("Scenarios that applied the buffer (deadline = calendar time - 30 min):")
    for sid in sorted(buffered):
        old = scen[sid]["constraints"]["ev_latest_finish"]
        say(f"  {sid}: calendar time {slot_time(buffered[sid])}, "
            f"scored deadline {slot_time(old)} -> rescored deadline {slot_time(buffered[sid])}")
    say("")

    # ---- 1. sanity: original constraints must reproduce the logged scores
    mismatch = 0
    for r in rows:
        orig = rescore(r, scen, scen[r["scenario"]]["constraints"])
        if correct(r["scenario"], orig) != correct(r["scenario"], r["score"] or {}):
            mismatch += 1
    say(f"Sanity check -- original scoring reproduced for "
        f"{len(rows) - mismatch}/{len(rows)} runs")
    if mismatch:
        raise SystemExit("Rescoring pipeline does not reproduce the logged "
                         "scores; stopping before writing anything.")
    say("")

    # ---- 2. does each rescored conflict still bind?
    say("Does the deadline conflict still bind after rescoring?")
    for sid in sorted(buffered):
        prices = list(archive.load_prices(scen[sid]["day"]))
        unc = cheapest_window(prices, EV_SLOTS)
        end = unc["start"] + EV_SLOTS
        old_lf, new_lf = scen[sid]["constraints"]["ev_latest_finish"], buffered[sid]
        tag = lambda lf: "binds" if end > lf else "does NOT bind"
        say(f"  {sid}: cheapest unconstrained EV window ends {slot_time(end)}; "
            f"old deadline {slot_time(old_lf)} {tag(old_lf)}, "
            f"new deadline {slot_time(new_lf)} {tag(new_lf)}")
    say("")

    # ---- 3. rescore
    out_rows, changed = [], []
    for r in rows:
        sid = r["scenario"]
        cons = copy.deepcopy(scen[sid]["constraints"])
        if sid in buffered:
            cons["ev_latest_finish"] = buffered[sid]
        new = rescore(r, scen, cons)
        before, after = correct(sid, r["score"] or {}), correct(sid, new)
        out_rows.append({"model": r["model"], "prompt_version": r["prompt_version"],
                         "scenario": sid, "before": before, "after": after})
        if before != after:
            ev = (r["committed"] or {}).get("ev_charger", {}).get("slot")
            changed.append({
                "run_id": r["run_id"], "scenario": sid, "model": r["model"],
                "prompt": r["prompt_version"],
                "ev_start": slot_time(ev) if ev is not None else "",
                "ev_finish": slot_time(ev + EV_SLOTS) if ev is not None else "",
                "old_deadline": slot_time(scen[sid]["constraints"]["ev_latest_finish"]),
                "new_deadline": slot_time(cons["ev_latest_finish"]),
                "before": before, "after": after})

    # ---- 4. outputs
    _, cells_b, overall_b = table5(out_rows, "before")
    agg_a, cells_a, overall_a = table5(out_rows, "after")

    say(f"Runs whose outcome changes: {len(changed)}")
    for k, n in sorted(collections.Counter(
            (c["scenario"], c["model"], f"{c['before']}->{c['after']}") for c in changed).items()):
        say(f"  {k[0]:4s} {k[1]:10s} {k[2]:12s} x{n}")
    say("")

    say("Table 5 (baseline / guided). Only cells that change are listed:")
    for m in MODELS:
        for f in FAMILIES:
            if cells_b[(m, f)] != cells_a[(m, f)]:
                say(f"  {m:10s} {f}:  {cells_b[(m, f)]}  ->  {cells_a[(m, f)]}")
    say("")

    say("Aggregate compliance per model (all 78 runs):")
    for m in MODELS:
        (cb, n), (ca, _) = overall_b[m], overall_a[m]
        say(f"  {m:10s} {cb}/{n} ({100*cb/n:.0f}%)  ->  {ca}/{n} ({100*ca/n:.0f}%)")
    say("")

    say("Full rescored Table 5 (baseline / guided):")
    say("  " + f"{'model':10s}" + "".join(f"{f:>14s}" for f in FAMILIES))
    for m in MODELS:
        say("  " + f"{m:10s}" + "".join(f"{cells_a[(m, f)]:>14s}" for f in FAMILIES))

    csv_path = OUT / "a1_exp2_main_rescored.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt", "family", "success_rate", "infeasibility_reported_rate"])
        for m in MODELS:
            for fam in FAMILIES:
                for p in PROMPTS:
                    v = agg_a[(m, p, fam)]
                    rate = f"{sum(v) / len(v):.4f}"
                    w.writerow([m, p, FAMKEY[fam], "", rate] if fam == "S4"
                               else [m, p, FAMKEY[fam], rate, ""])

    ch_path = OUT / "a1_changed_runs.csv"
    with open(ch_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(changed[0].keys()) if changed else ["run_id"])
        w.writeheader()
        w.writerows(changed)

    (OUT / "a1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {csv_path.relative_to(ROOT)}, {ch_path.relative_to(ROOT)}, "
          f"revision/outputs/a1_summary.txt")


if __name__ == "__main__":
    main()
