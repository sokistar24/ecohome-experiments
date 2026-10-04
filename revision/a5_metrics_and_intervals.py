"""
Phase A, step A5 -- Metric definitions, absolute cost gaps and uncertainty intervals.

Why
---
Reviewer 3 asked for (i) explicit definitions of success, optimality,
near-optimality, gamma and $/success before Table 4, with tolerances;
(ii) how gamma behaves when the MILP cost is near zero or negative, and
absolute GBP gaps beside percentages; (iii) uncertainty intervals that
respect the repeated-run structure; (iv) paired comparisons; (v) a fuller
description of how the evaluation days were sampled. Reviewer 1 asked for
the $/success definition. This step answers all of these from the logs.

Sampling unit
-------------
Repeats run at temperature 0, so they are strongly correlated: the script
reports how often all three repeats agree. Intervals therefore treat the
DAY (Exp 1) or the SCENARIO x PROMPT cell (Exp 2) as the sampling unit, not
the run. Proportions use Wilson score intervals on that unit; paired and
continuous comparisons use a bootstrap that resamples days.

No LLM calls. Run from the repo root (after A1 and A2):
    python -m revision.a5_metrics_and_intervals

Outputs (revision/outputs/):
    a5_metric_definitions.md   definitions and sampling text for the Methods section
    a5_table4_extended.csv     Table 4 plus corrected near-optimality, GBP gaps, intervals
    a5_table4.tex              LaTeX rows for the revised Table 4
    a5_intervals.csv           every interval reported
    a5_summary.txt             console summary
"""
from __future__ import annotations

import collections
import copy
import csv
import json
import math
import random
import statistics as st

from experiments import archive, config
from revision.a1_rescore_departure_buffer import (
    load_scenarios, detect_buffered, rescore, correct, family)
from revision.a2_exp3_milp_reference import milp_arms

ROOT = config.ROOT
RUNS = ROOT / "data" / "runs"
EXP1_MAIN = ROOT / "data" / "results" / "exp1_main.csv"
OUT = ROOT / "revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

Z = 1.959964
B = 10000
SEED = 20261002
MODELS5 = ["gpt", "gemini", "claude", "llama-3.3", "qwen-3"]
NEAR_OPT = 0.01
OPT_TOL = 1e-6


def wilson(p: float, n: float):
    """Wilson score interval; p may come from fractional successes."""
    if n <= 0:
        return (None, None)
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot(units: list, stat, seed=SEED):
    """Percentile bootstrap over units (each unit kept intact)."""
    rng = random.Random(seed)
    vals = sorted(stat([rng.choice(units) for _ in units]) for _ in range(B))
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def fmt_ci(lo, hi, pct=False):
    if lo is None:
        return "--"
    return f"[{100*lo:.0f}, {100*hi:.0f}]" if pct else f"[{lo:+.3f}, {hi:+.3f}]"


def main():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("A5 -- metric definitions, absolute gaps and intervals")
    say("")
    intervals = []

    # ============================================================ Exp 1
    rows = [json.loads(l) for l in open(RUNS / "exp1.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["scenario"] == "multi" and r["score"]]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["interface"])].append(r)

    agree = collections.defaultdict(set)
    for r in rows:
        agree[(r["model"], r["interface"], r["day"])].add(r["score"]["success"])
    same = sum(len(v) == 1 for v in agree.values())
    say(f"Repeat agreement, Exp 1: all three repeats share the same success outcome "
        f"in {same}/{len(agree)} model-interface-day cells ({100*same/len(agree):.0f}%)")

    published = {(x["model"], x["interface"]): x for x in csv.DictReader(open(EXP1_MAIN))}
    t4 = []
    for (m, i), rs in sorted(cells.items(), key=lambda kv: (MODELS5.index(kv[0][0]), kv[0][1])):
        sc = [r["score"] for r in rs]
        succ = [s["success"] for s in sc]
        opt = [s["optimal"] for s in sc]
        near_pub = [s["cost_gap"] is not None and s["cost_gap"] <= NEAR_OPT for s in sc]
        near_ok = [s["success"] and s["cost_gap"] is not None and s["cost_gap"] <= NEAR_OPT
                   for s in sc]
        ok = [s for s in sc if s["success"] and s["cost_gap"] is not None]
        gaps = [s["cost_gap"] for s in ok]
        dj = [s["agent_cost"] - s["milp_cost"] for s in ok]
        cost_ps = sum(r["est_cost_usd"] or 0 for r in rs) / max(1, sum(succ))

        pub = published[(m, i)]
        for got, key in ((mean(succ), "success_rate"), (mean(opt), "optimality_rate"),
                         (mean(near_pub), "near_optimal_rate")):
            assert abs(got - float(pub[key])) < 1e-3, f"{m} {i} {key} does not reproduce"
        assert abs(100 * st.median(gaps) - float(pub["median_cost_gap_pct"])) < 1e-2

        # day-level units for intervals
        by_day = collections.defaultdict(list)
        for r in rs:
            by_day[r["day"]].append(r["score"])
        days = sorted(by_day)
        s_day = [mean([s["success"] for s in by_day[d]]) for d in days]
        o_day = [mean([s["optimal"] for s in by_day[d]]) for d in days]
        s_ci, o_ci = wilson(mean(s_day), len(days)), wilson(mean(o_day), len(days))
        t4.append({"model": m, "interface": i, "runs": len(rs), "days": len(days),
                   "success": round(mean(succ), 3), "success_ci": fmt_ci(*s_ci, pct=True),
                   "optimal": round(mean(opt), 3), "optimal_ci": fmt_ci(*o_ci, pct=True),
                   "near_opt_published": round(mean(near_pub), 3),
                   "near_opt_successful_only": round(mean(near_ok), 3),
                   "median_gap_pct": round(100 * st.median(gaps), 2),
                   "max_gap_pct": round(100 * max(gaps), 2),
                   "median_gap_gbp": round(st.median(dj), 3),
                   "max_gap_gbp": round(max(dj), 3),
                   "usd_per_success": round(cost_ps, 4)})
        intervals += [{"quantity": f"Exp1 success {m} {i}", "estimate": round(mean(succ), 3),
                       "ci_low": round(s_ci[0], 3), "ci_high": round(s_ci[1], 3),
                       "method": f"Wilson, n = {len(days)} days"},
                      {"quantity": f"Exp1 optimality {m} {i}", "estimate": round(mean(opt), 3),
                       "ci_low": round(o_ci[0], 3), "ci_high": round(o_ci[1], 3),
                       "method": f"Wilson, n = {len(days)} days"}]
    say("Sanity check -- success, optimality, near-optimality and median gap reproduce "
        "exp1_main.csv for all 10 cells")
    say("")

    jstar = {}
    for r in rows:
        jstar[r["day"]] = r["score"]["milp_cost"]
    say(f"MILP reference cost J* across the 12 Exp 1 days: GBP {min(jstar.values()):.2f} "
        f"to {max(jstar.values()):.2f} (never zero or negative); smallest on "
        f"{min(jstar, key=jstar.get)}")
    top = max((r for r in rows if r["score"]["success"] and r["score"]["cost_gap"] is not None),
              key=lambda r: r["score"]["cost_gap"])
    s = top["score"]
    say(f"Largest gap: {top['model']} {top['interface']} on {top['day']}: "
        f"{100*s['cost_gap']:.2f}% = GBP {s['agent_cost'] - s['milp_cost']:.2f} "
        f"(J* = {s['milp_cost']:.2f}, agent {s['agent_cost']:.2f})")
    say("")

    say("Table 4 extended (intervals: Wilson, unit = day):")
    say(f"  {'model':10s} {'if':4s} {'success':>8s} {'95% CI':>10s} {'optimal':>8s} {'95% CI':>10s} "
        f"{'near-opt pub':>12s} {'near-opt fixed':>14s} {'med £':>7s} {'max £':>6s} {'max %':>7s}")
    for t in t4:
        say(f"  {t['model']:10s} {t['interface']:4s} {t['success']:8.2f} {t['success_ci']:>10s} "
            f"{t['optimal']:8.2f} {t['optimal_ci']:>10s} {t['near_opt_published']:12.2f} "
            f"{t['near_opt_successful_only']:14.2f} {t['median_gap_gbp']:7.3f} "
            f"{t['max_gap_gbp']:6.2f} {t['max_gap_pct']:7.2f}")
    changed = [t for t in t4 if abs(t["near_opt_published"] - t["near_opt_successful_only"]) > 1e-9]
    say(f"  Near-optimality changes where constraint-violating runs had been counted: "
        + ", ".join(f"{t['model']} {t['interface']} {t['near_opt_published']:.2f} -> "
                    f"{t['near_opt_successful_only']:.2f}" for t in changed))
    say("")

    say("Function calling minus text-parsed actions (same days and repeats; bootstrap over days):")
    say(f"  {'model':10s} {'success, pp':>12s} {'95% CI':>16s} {'optimality, pp':>15s} {'95% CI':>16s}")
    for m in MODELS5:
        parts = []
        for key in ("success", "optimal"):
            by_day = collections.defaultdict(lambda: {"fc": [], "text": []})
            for r in rows:
                if r["model"] == m:
                    by_day[r["day"]][r["interface"]].append(r["score"][key])
            units = [mean(v["fc"]) - mean(v["text"]) for v in by_day.values()]
            lo, hi = boot(units, lambda u: mean(u))
            est = mean(units)
            parts.append(f"{100*est:+12.1f} {'[' + f'{100*lo:+.1f}, {100*hi:+.1f}' + ']':>16s}")
            intervals.append({"quantity": f"Exp1 {key} FC - text {m}", "estimate": round(est, 3),
                              "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                              "method": "paired bootstrap over 12 days"})
        say(f"  {m:10s} " + " ".join(parts))
    say("")

    # ============================================================ Exp 2
    scen = load_scenarios()
    buffered = detect_buffered(scen)
    rows2 = [json.loads(l) for l in open(RUNS / "exp2.jsonl", encoding="utf-8")]
    cell2 = collections.defaultdict(list)
    gpt_inv = []
    for r in rows2:
        cons = copy.deepcopy(scen[r["scenario"]]["constraints"])
        if r["scenario"] in buffered:
            cons["ev_latest_finish"] = buffered[r["scenario"]]
        sc = rescore(r, scen, cons)
        cell2[(r["model"], r["scenario"], r["prompt_version"])].append(correct(r["scenario"], sc))
        if (r["model"] == "gpt" and family(r["scenario"]) != "S4"
                and sc.get("constraint_violations") and sc.get("agent_cost") is not None):
            gpt_inv.append((sc["agent_cost"], sc["milp_cost"]))
    same2 = sum(len(set(v)) == 1 for v in cell2.values())
    say(f"Repeat agreement, Exp 2: {same2}/{len(cell2)} model-scenario-prompt cells "
        f"({100*same2/len(cell2):.0f}%) have identical outcomes across repeats")
    say("Aggregate compliance (A1 rule), Wilson interval with unit = scenario x prompt (26):")
    for m in MODELS5:
        units = [mean(v) for (mm, _, _), v in cell2.items() if mm == m]
        est = mean(units)
        lo, hi = wilson(est, len(units))
        say(f"  {m:10s} {100*est:5.1f}%  95% CI [{100*lo:.0f}, {100*hi:.0f}]")
        intervals.append({"quantity": f"Exp2 compliance {m}", "estimate": round(est, 3),
                          "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                          "method": f"Wilson, n = {len(units)} scenario x prompt cells"})
    say("")
    if gpt_inv:
        dj = [a - b for a, b in gpt_inv]
        say(f"GPT constraint-violating schedules on feasible scenarios ({len(gpt_inv)} runs): "
            f"mean GBP {mean(dj):+.2f} vs the feasible optimum "
            f"(mean J* = GBP {mean([b for _, b in gpt_inv]):.2f}). Replaces the -152.9% figure.")
        say("")

    # ============================================================ Exp 3
    sel = archive.load_selection()
    regime_of = {d: rg for rg, ds in sel["exp3_regimes"].items() for d in ds}
    runs3 = collections.defaultdict(list)
    for l in open(RUNS / "exp3.jsonl", encoding="utf-8"):
        r = json.loads(l)
        c = (r["score"] or {}).get("net_cost_realized")
        if c is not None:
            runs3[(r["model"], r["day"], r["scenario"])].append(c)
    per_day = collections.defaultdict(list)
    for d in regime_of:
        for m in ("gpt", "gemini", "claude"):
            if runs3[(m, d, "price_only")] and runs3[(m, d, "weather_aware")]:
                per_day[d].append(mean(runs3[(m, d, "price_only")]) -
                                  mean(runs3[(m, d, "weather_aware")]))
    milp_val = {}
    for d in regime_of:
        a = milp_arms(d)
        milp_val[d] = a["po_milp"]["cost"] - a["wa_milp"]["cost"]
    units = [(mean(per_day[d]), milp_val[d]) for d in sorted(regime_of) if per_day[d]]
    a_est = mean([u[0] for u in units]); m_est = mean([u[1] for u in units])
    a_ci = boot(units, lambda u: mean([x[0] for x in u]))
    m_ci = boot(units, lambda u: mean([x[1] for x in u]))
    g_ci = boot(units, lambda u: mean([x[1] - x[0] for x in u]))
    say("Exp 3, value of weather information pooled over all 15 days "
        "(GBP/day, positive = cheaper; bootstrap over days):")
    say(f"  weather-aware MILP vs price-only MILP   {m_est:+.3f}  95% CI {fmt_ci(*m_ci)}")
    say(f"  weather-aware vs price-only agents      {a_est:+.3f}  95% CI {fmt_ci(*a_ci)}")
    say(f"  difference (value the agents leave)     {m_est - a_est:+.3f}  95% CI {fmt_ci(*g_ci)}")
    intervals += [
        {"quantity": "Exp3 weather value, MILP", "estimate": round(m_est, 3),
         "ci_low": round(m_ci[0], 3), "ci_high": round(m_ci[1], 3), "method": "bootstrap over 15 days"},
        {"quantity": "Exp3 weather value, agents (matched)", "estimate": round(a_est, 3),
         "ci_low": round(a_ci[0], 3), "ci_high": round(a_ci[1], 3), "method": "bootstrap over 15 days"},
        {"quantity": "Exp3 MILP minus agents", "estimate": round(m_est - a_est, 3),
         "ci_low": round(g_ci[0], 3), "ci_high": round(g_ci[1], 3), "method": "bootstrap over 15 days"}]
    say("")

    # ============================================================ sampling
    stats = sel["day_stats"]
    complete = sorted(d for d, s_ in stats.items()
                      if s_["n_slots"] == config.SLOTS_PER_DAY and s_["cov"] is not None)
    ranked = sorted(complete, key=lambda d: stats[d]["cov"])
    n = len(ranked)
    b1, b2 = stats[ranked[n // 3]]["cov"], stats[ranked[2 * n // 3]]["cov"]
    say("Sampling (from day_selection.json and fetch_archives.py):")
    say(f"  window {sel['window'][0]} to {sel['window'][1]}; {n} complete Agile days "
        f"(48 slots) are candidates")
    say(f"  Exp 1: ranked by coefficient of variation of import price; terciles split at "
        f"CoV = {b1:.3f} and {b2:.3f}; 4 days per tercile, evenly spaced in time; "
        f"a negative-price day is swapped in if none is selected")
    say("  Exp 3: forecast daytime (06-20 h) mean cloud cover < 30% sunny, > 70% overcast, "
        "otherwise mixed; 5 days per regime, evenly spaced in time")
    say("  Exp 4a: first Monday-Sunday week fully covered by archived data")
    if abs(sel.get("export_rate_gbp", config.EXPORT_RATE_GBP) - config.EXPORT_RATE_GBP) > 1e-9:
        say(f"  Note: day_selection.json records export_rate_gbp = {sel['export_rate_gbp']}, "
            f"but all scoring uses config.EXPORT_RATE_GBP = {config.EXPORT_RATE_GBP}; "
            f"day selection does not use the export rate")

    # ============================================================ write
    defs = f"""# Metric definitions (from experiments/scorer.py and analyze.py)

All definitions apply per run. J* is the MILP optimum for the same day, appliances
and constraints; J is the net cost of the committed schedule under the same
accounting.

- **Success**: every requested appliance is committed through the scheduling tool and
  the committed schedule violates no stated constraint (horizon, EV deadline,
  power cap, quiet hours). For infeasible requests (S4), success means reporting
  infeasibility without committing the impossible appliance.
- **Exact optimality**: all appliances committed and |J - J*| < GBP {OPT_TOL:g}
  (no run in Exp 1 is optimal without also being successful).
- **Near-optimality**: success and gamma <= {NEAR_OPT:.0%}. (The submitted version did not
  require success, which counted 12 constraint-violating text-interface runs as
  near-optimal; see Table 4.)
- **gamma (relative cost gap)**: (J - J*) / |J*|, undefined when |J*| < 1e-9. In
  Exp 1, J* ranges from GBP {min(jstar.values()):.2f} to {max(jstar.values()):.2f}, so gamma is always
  defined. The absolute gap J - J* (GBP) is reported beside it because gamma
  inflates on low-cost days (the largest gamma, {100*s['cost_gap']:.1f}%, is GBP {s['agent_cost'] - s['milp_cost']:.2f}).
  Median and maximum gaps are over successful runs.
- **$/success**: total estimated API inference cost of all runs in a cell (successful
  or not, at list prices) divided by the number of successful runs.
- **Deadline**: an EV deadline of HH:MM is met when charging finishes at or before
  HH:MM, whether the time is stated directly or taken from a calendar entry.

Uncertainty. Repeats run at temperature 0 and are strongly correlated ({100*same/len(agree):.0f}% of Exp 1
cells and {100*same2/len(cell2):.0f}% of Exp 2 cells have identical outcomes across the three repeats).
Intervals therefore use the day (Exp 1) or scenario x prompt cell (Exp 2) as the sampling
unit: Wilson score intervals for proportions, and a percentile bootstrap over days
({B} resamples) for paired and continuous comparisons.

Sampling. Candidate days are the {n} complete Octopus Agile days (48 half-hourly prices,
region C) from {sel['window'][0]} to {sel['window'][1]}. Exp 1 ranks them by the coefficient of variation of
the import price and splits them into terciles at CoV = {b1:.3f} and {b2:.3f}; four days per tercile
are taken evenly spaced in time, with a negative-price day swapped in if none is selected.
Exp 3 classifies days by forecast daytime (06-20 h) mean cloud cover: below 30% sunny, above
70% overcast, otherwise mixed; five days per regime, evenly spaced in time. Exp 4a uses the
first Monday-Sunday week fully covered by the archive.
"""
    (OUT / "a5_metric_definitions.md").write_text(defs, encoding="utf-8")
    with open(OUT / "a5_table4_extended.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(t4[0].keys()))
        w.writeheader(); w.writerows(t4)
    with open(OUT / "a5_intervals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(intervals[0].keys()))
        w.writeheader(); w.writerows(intervals)
    tex = ["% Revised Table 4 body. Columns: model | interface | success [95% CI] |",
           "% optimal [95% CI] | near-optimal | median gap GBP | max gap GBP (max %) | $/success"]
    for t in t4:
        tex.append(f"{t['model']} & {t['interface']} & {t['success']:.2f} {t['success_ci']} & "
                   f"{t['optimal']:.2f} {t['optimal_ci']} & {t['near_opt_successful_only']:.2f} & "
                   f"{t['median_gap_gbp']:.3f} & {t['max_gap_gbp']:.2f} ({t['max_gap_pct']:.1f}\\%) & "
                   f"{t['usd_per_success']:.4f} " + r"\\")
    (OUT / "a5_table4.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (OUT / "a5_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote revision/outputs/a5_metric_definitions.md, a5_table4_extended.csv, "
          "a5_table4.tex, a5_intervals.csv, a5_summary.txt")


if __name__ == "__main__":
    main()
