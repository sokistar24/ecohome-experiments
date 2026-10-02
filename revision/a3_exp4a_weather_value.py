"""
Phase A, step A3 -- Seven-day rolling deployment (Exp 4a): where the savings come from.

Why
---
Table 7 reports that the agents capture 96.7-98.0% of the savings between an
off-peak timer and the extended (weather-aware) MILP oracle. That share mixes
two things: shifting load to cheap prices, and using the PV forecast. The
price-only MILP alone already captures 98.1%, so the share cannot show
whether the agents use the forecast at all. This script separates the two and
adds the checks a reader needs to interpret Table 7:

  1. Reproduces the published Table 7 (exp4_main.csv) as a sanity check.
  2. Weather value captured per policy:
         omega_p = 100 * (J_price-only-MILP - J_p) / (J_price-only-MILP - J_oracle)
     0% = no better than ignoring PV, 100% = matches the weather-aware MILP.
  3. Perfect-foresight weekly bound (MILP deciding on actual PV).
  4. Feasibility of each rule-based baseline (EV deadline met on how many days).
  5. Agent non-commits (model-days without a full schedule).
  6. Schedule-match diagnostic, for Exp 4a and the Exp 3 weather-aware arm:
     how often each agent's committed schedule is exactly the price-only MILP
     schedule, exactly the weather-aware MILP schedule, or neither.

No LLM calls. Run from the repo root (after A2):
    python -m revision.a3_exp4a_weather_value

Outputs (revision/outputs/):
    a3_table7.csv / a3_table7.tex   revised Table 7
    a3_schedule_match.csv           schedule-match counts by experiment and model
    a3_summary.txt                  console summary
"""
from __future__ import annotations

import collections
import csv
import json
import statistics as st

from experiments import config
from revision.a2_exp3_milp_reference import milp_arms

ROOT = config.ROOT
RUNS4 = ROOT / "data" / "runs" / "exp4a.jsonl"
RUNS3 = ROOT / "data" / "runs" / "exp3.jsonl"
BASELINES = ROOT / "data" / "results" / "baselines.json"
EXP4_MAIN = ROOT / "data" / "results" / "exp4_main.csv"
OUT = ROOT / "revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt", "gemini", "claude"]
MODEL_LABEL = {"gpt": "GPT-4o-mini agent", "gemini": "Gemini 2.5 Flash agent",
               "claude": "Claude Sonnet 4.6 agent"}
RULES = {"immediate": "Immediate start", "off_peak_timer": "Off-peak timer",
         "greedy_slot": "Greedy cheapest-slot heuristic"}
TOL = 1e-2          # published week costs are rounded to 3 dp


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def main():
    base = json.loads(BASELINES.read_text(encoding="utf-8"))["per_day"]
    rows4 = [json.loads(l) for l in open(RUNS4, encoding="utf-8")]
    days = sorted({r["day"] for r in rows4})
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"A3 -- Exp 4a seven-day deployment ({days[0]} to {days[-1]})")
    say("")

    # ------------------------------------------------ weekly costs per policy
    week = {}
    for key in list(RULES) + ["price_only_milp", "oracle"]:
        week[key] = sum(base[d]["wfh_ev18"][key]["net_cost_realized"] for d in days)
    pf = {d: milp_arms(d)["pf_milp"]["cost"] for d in days}
    week["perfect_foresight"] = sum(pf.values())

    per = collections.defaultdict(list)
    attempts, noncommit = collections.Counter(), collections.Counter()
    for r in rows4:
        attempts[r["model"]] += 1
        c = (r["score"] or {}).get("net_cost_realized")
        if c is None:
            noncommit[r["model"]] += 1
        per[(r["model"], r["day"])].append(c)
    for m in MODELS:   # same aggregation as analyze_exp4: per-day mean, summed
        week[m] = sum(mean(per[(m, d)]) for d in days)
    agent_viol_days = collections.defaultdict(set)
    for r in rows4:
        if (r["score"] or {}).get("constraint_violations"):
            agent_viol_days[r["model"]].add(r["day"])

    published = {x["policy"]: float(x["week_cost"]) for x in csv.DictReader(open(EXP4_MAIN))}
    for key, pub in (("immediate", "immediate"), ("off_peak_timer", "off_peak_timer"),
                     ("greedy_slot", "greedy_slot"), ("price_only_milp", "price_only_milp"),
                     ("oracle", "oracle"), ("gpt", "agent_gpt"),
                     ("gemini", "agent_gemini"), ("claude", "agent_claude")):
        assert abs(week[key] - published[pub]) < TOL, f"{key} does not reproduce exp4_main.csv"
    say("Sanity check -- all eight published Table 7 costs reproduced")
    say("")

    say("Agent non-commits (passes with no full schedule; that day averages the other passes):")
    for m in MODELS:
        say(f"  {m:7s} {noncommit[m]}/{attempts[m]}")
    say("")

    say("Rule-based baselines -- EV deadline (18:00) met on how many of the 7 days:")
    feas = {}
    for key, label in RULES.items():
        ok = sum(not base[d]["wfh_ev18"][key]["violations"] for d in days)
        feas[key] = ok
        say(f"  {label:32s} {ok}/7")
    say("")

    # ------------------------------------------------ shares
    timer, oracle, pom = week["off_peak_timer"], week["oracle"], week["price_only_milp"]
    eta = lambda j: 100 * (timer - j) / (timer - oracle)
    omega = lambda j: 100 * (pom - j) / (pom - oracle)
    order = [("immediate", RULES["immediate"]), ("off_peak_timer", RULES["off_peak_timer"]),
             ("greedy_slot", RULES["greedy_slot"]), ("price_only_milp", "Price-only MILP"),
             ("oracle", "Weather-aware MILP (oracle)"),
             ("perfect_foresight", "Perfect-foresight MILP")] + \
            [(m, MODEL_LABEL[m]) for m in MODELS]
    table = []
    for key, label in order:
        j = week[key]
        table.append({
            "policy": label, "week_cost_gbp": round(j, 2),
            "ev_deadline_met_days": (f"{feas[key]}/7" if key in feas else
                                     f"{7 - len(agent_viol_days[key])}/7" if key in MODELS
                                     else "7/7"),          # MILPs: feasible by construction
            "timer_to_oracle_share_pct": round(eta(j), 1),
            "weather_value_captured_pct": (round(omega(j), 1)
                                           if key not in RULES else ""),
            "non_commit": f"{noncommit[key]}/{attempts[key]}" if key in MODELS else ""})

    say(f"Weather value available in the week: price-only MILP {pom:.2f} - "
        f"oracle {oracle:.2f} = GBP {pom - oracle:.2f} "
        f"(perfect foresight: GBP {pom - week['perfect_foresight']:.2f})")
    say("")
    say(f"  {'policy':32s} {'7-day £':>8s} {'deadline':>9s} {'timer->oracle %':>16s} "
        f"{'weather value %':>16s} {'non-commit':>10s}")
    for t in table:
        say(f"  {t['policy']:32s} {t['week_cost_gbp']:8.2f} {t['ev_deadline_met_days']:>9s} "
            f"{t['timer_to_oracle_share_pct']:16.1f} {str(t['weather_value_captured_pct']):>16s} "
            f"{t['non_commit']:>10s}")
    say("")

    # ------------------------------------------------ schedule-match diagnostic
    say("Schedule match: committed schedule vs the two MILP schedules for that day")
    say("  ('identical' = the two MILPs choose the same schedule that day)")
    match_rows = []
    for exp, path, arm in (("exp4a", RUNS4, None), ("exp3 weather-aware", RUNS3, "weather_aware")):
        cnt = collections.defaultdict(collections.Counter)
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if arm and r["scenario"] != arm:
                continue
            s = r["score"] or {}
            if s.get("net_cost_realized") is None:
                continue
            got = s["committed_starts"]
            po = base[r["day"]]["wfh_ev18"]["price_only_milp"]["starts"]
            ox = base[r["day"]]["wfh_ev18"]["oracle"]["starts"]
            if po == ox:
                k = "identical" if got == po else "other"
            else:
                k = ("price_only_milp" if got == po else
                     "weather_milp" if got == ox else "other")
            cnt[r["model"]][k] += 1
        say(f"  {exp}:")
        for m in MODELS:
            c = cnt[m]
            n = sum(c.values())
            say(f"    {m:7s} n={n:3d}  price-only MILP {c['price_only_milp']:3d}  "
                f"weather MILP {c['weather_milp']:3d}  identical {c['identical']:3d}  "
                f"other {c['other']:3d}")
            match_rows.append({"experiment": exp, "model": m, "n": n,
                               "price_only_milp": c["price_only_milp"],
                               "weather_milp": c["weather_milp"],
                               "identical": c["identical"], "other": c["other"]})

    # ------------------------------------------------ write
    with open(OUT / "a3_table7.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader(); w.writerows(table)
    with open(OUT / "a3_schedule_match.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(match_rows[0].keys()))
        w.writeheader(); w.writerows(match_rows)
    tex = ["% Revised Table 7 body. Columns: policy | seven-day cost (GBP) | EV deadline met |",
           "% timer-to-oracle savings captured (%) | weather value captured (%)"]
    for t in table:
        wv = f"{t['weather_value_captured_pct']:.1f}" if t["weather_value_captured_pct"] != "" else "--"
        tex.append(f"{t['policy']} & {t['week_cost_gbp']:.2f} & {t['ev_deadline_met_days']} & "
                   f"{t['timer_to_oracle_share_pct']:.1f} & {wv} " + r"\\")
    (OUT / "a3_table7.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (OUT / "a3_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote revision/outputs/a3_table7.csv, a3_table7.tex, "
          "a3_schedule_match.csv, a3_summary.txt")


if __name__ == "__main__":
    main()
