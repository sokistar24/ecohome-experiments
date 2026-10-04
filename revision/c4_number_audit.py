"""
Phase C, step 4 -- audit every quantitative claim in the revised paper.

Each check below quotes a claim as written in article.tex and recomputes it
from revision/outputs/ (Phases A-C) or directly from the run logs. Run after
`python -m revision.c0_paper_assets`, `c1_coupling_check` and `c2_drift_check`.

    python -m revision.c4_number_audit

Prints PASS/FAIL per claim and exits non-zero if any claim fails.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys

from experiments import config

OUT = config.ROOT / "revision" / "outputs"
RUNS = config.RUNS
results = []


def rows(name):
    return list(csv.DictReader(open(OUT / name, encoding="utf-8")))


def logs(name):
    return [json.loads(l) for l in open(RUNS / f"{name}.jsonl", encoding="utf-8")]


def check(claim, ok, got=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {claim}" + (f"   [{got}]" if got else ""))


def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def main():
    # ---------------------------------------------------------------- Exp 1
    t4 = {(r["model"], r["interface"]): r for r in rows("a5_table4_extended.csv")}
    fc = [t4[(m, "fc")] for m in ("gpt", "gemini", "claude", "llama-3.3", "qwen-3")]
    check("E1: every model >=97% success under FC", all(float(r["success"]) >= 0.97 for r in fc))
    check("E1: GPT and Gemini exact optimum in every FC run",
          float(t4[("gpt", "fc")]["optimal"]) == 1 and float(t4[("gemini", "fc")]["optimal"]) == 1)
    opt = {m: round(100 * float(t4[(m, "fc")]["optimal"])) for m in ("claude", "qwen-3", "llama-3.3")}
    check("E1: exact optimality Claude 86%, Qwen 72%, Llama 36%",
          opt == {"claude": 86, "qwen-3": 72, "llama-3.3": 36}, opt)
    check("E1: median FC cost gap GBP 0.00 for every model",
          all(abs(float(r["median_gap_gbp"])) < 0.005 for r in fc))
    check("E1: largest FC gap GBP 1.86 (Qwen, 122.9%)",
          near(t4[("qwen-3", "fc")]["max_gap_gbp"], 1.86, 0.005)
          and near(t4[("qwen-3", "fc")]["max_gap_pct"], 122.86, 0.05))
    iv = {r["quantity"]: r for r in rows("a5_intervals.csv")}

    def ivs(q):
        r = iv[q]
        return round(100 * float(r["estimate"])), round(100 * float(r["ci_low"])), round(100 * float(r["ci_high"]))
    exp = {"gpt": (69, 53, 83), "gemini": (36, 14, 61), "claude": (53, 31, 78),
           "llama-3.3": (6, -14, 25), "qwen-3": (25, 0, 53)}
    got = {m: ivs(f"Exp1 optimal FC - text {m}") for m in exp}
    check("E1: FC-text optimality differences and CIs", got == exp, got)
    check("E1: Claude text success 50%", near(t4[("claude", "text")]["success"], 0.5, 0.01))
    e1 = logs("exp1")
    cl = [r for r in e1 if r["model"] == "claude" and r["interface"] == "text"
          and r["scenario"] == "single" and not (r["score"] or {}).get("success")]
    check("E1: all 32 failed Claude text single-appliance runs made no tool call",
          len(cl) == 32 and all(not r["tool_calls"] for r in cl), len(cl))
    pub = {(r["model"], r["interface"]): float(r["mean_tokens"])
           for r in csv.DictReader(open(config.DATA / "results" / "exp1_main.csv", encoding="utf-8"))}
    ratios = [pub[(m, "text")] / pub[(m, "fc")] for m in ("gpt", "gemini", "claude", "llama-3.3", "qwen-3")]
    check("E1: text uses 1.1 to 8.3 times the tokens of FC",
          near(min(ratios), 1.1, 0.05) and near(max(ratios), 8.3, 0.05),
          f"{min(ratios):.2f}-{max(ratios):.2f}")

    # ---------------------------------------------------------------- coupling
    txt = (OUT / "c1_coupling.txt").read_text(encoding="utf-8")
    check("Methods: decoupled on 12/12 Exp 1 days", "12/12 days" in txt)
    check("Methods: Exp 3 coupled on 15/15 days, mean GBP 0.16, cap broken 14/15",
          "15/15 days" in txt and "mean GBP 0.16/day" in txt and "14/15 days" in txt)

    # ---------------------------------------------------------------- Exp 2 direct
    exp = {"claude": (100, 87, 100), "gemini": (94, 78, 98), "qwen-3": (72, 53, 85),
           "gpt": (67, 47, 82), "llama-3.3": (50, 32, 68)}
    got = {m: ivs(f"Exp2 compliance {m}") for m in exp}
    check("E2: direct compliance and CIs (both prompts)", got == exp, got)
    sev = {r["model"]: r for r in rows("a4_severity_by_model.csv") if r["prompt"] == "both"}
    inv = sum(int(r["invalid_commits"]) for r in sev.values())
    com = sum(int(r["committing_runs"]) for r in sev.values())
    check("E2: 52 of 343 committing runs invalid", (inv, com) == (52, 343), (inv, com))
    over = [float(r["max_overrun_h"]) for r in sev.values() if int(r["deadline_violations"])]
    inv_runs = rows("a4_invalid_runs.csv")
    lates = [int(r["deadline_overrun_min"]) / 60 for r in inv_runs
             if int(r["deadline_overrun_min"]) > 0 and not r["scenario"].startswith("S4")]
    check("E2: every missed deadline on feasible scenarios 7.5-9.5 h late",
          min(lates) == 7.5 and max(lates) == 9.5, f"{min(lates)}-{max(lates)}")
    caps = sorted(float(r["cap_excess_kw"]) for r in inv_runs
                  if float(r["cap_excess_kw"]) > 0 and not r["scenario"].startswith("S4"))
    check("E2: all cap violations 2.2 kW except one", caps.count(2.2) == len(caps) - 1, caps)
    fab = {m: int(sev[m]["s4_fabricated"]) for m in sev}
    check("E2: fabrications Qwen 7, Llama 6, GPT 3, Claude 0, Gemini 0",
          fab == {"gpt": 3, "gemini": 0, "claude": 0, "llama-3.3": 6, "qwen-3": 7}, fab)

    # ---------------------------------------------------------------- Exp 2 arms
    arms = {r["model"]: r for r in rows("b3_exp2_arms.csv")}
    exp = {"gpt": (72, 87, 82), "gemini": (97, 97, 100), "claude": (100, 100, 100),
           "llama-3.3": (38, 59, 90), "qwen-3": (79, 64, 64)}
    got = {m: tuple(round(100 * float(arms[m][f"{a}_rate"])) for a in ("direct", "guard", "hybrid"))
           for m in exp}
    check("E2: direct/guard/hybrid rates", got == exp, got)
    check("E2: Llama +51 [31, 69], GPT +10 [-13, 33], Qwen -15 [-41, 10]",
          (round(float(arms["llama-3.3"]["hybrid_minus_direct_pp"])), arms["llama-3.3"]["diff_ci"]) == (51, "[+31, +69]")
          and (round(float(arms["gpt"]["hybrid_minus_direct_pp"])), arms["gpt"]["diff_ci"]) == (10, "[-13, +33]")
          and (round(float(arms["qwen-3"]["hybrid_minus_direct_pp"])), arms["qwen-3"]["diff_ci"]) == (-15, "[-41, +10]"))
    b3 = (OUT / "b3_summary.txt").read_text(encoding="utf-8")
    fixed = sum(int(x) for x in __import__("re").findall(r"fixed by hybrid\s+(\d+)", b3))
    broken = sum(int(x) for x in __import__("re").findall(r"broken by hybrid\s+(\d+)", b3))
    check("E2: hybrid fixes 32 and introduces 13 failures", (fixed, broken) == (32, 13), (fixed, broken))
    fails = rows("b3_hybrid_failures.csv")
    n_int = sum(r["cause"].startswith("interpretation") for r in fails)
    check("E2: 25 hybrid failures, 22 interpretation, 3 protocol",
          (len(fails), n_int) == (25, 22), (len(fails), n_int))
    ext = {r["model"]: r for r in rows("b3_extraction.csv")}
    ok_ev = sum(int(r["ev_deadline_exact"].split("/")[0]) for r in ext.values())
    n_ev = sum(int(r["ev_deadline_exact"].split("/")[1]) for r in ext.values())
    check("E2: EV deadline extracted exactly in 166 of 178 runs", (ok_ev, n_ev) == (166, 178), (ok_ev, n_ev))
    check("E2: Qwen invents a 4 kW cap in 7 runs",
          ext["qwen-3"]["spurious_cap"].startswith("7/") and ext["qwen-3"]["spurious_cap_values_kw"] == "4.0")
    check("E2: guard for Qwen blocks 13, breaks 10, fixes 4",
          "qwen-3     pass 24  repair  2  block 13  -> fixed  4, broke 10" in b3)
    tok_h = [int(arms[m]["hybrid_tokens"]) for m in arms]
    tok_d = [int(arms[m]["direct_tokens"]) for m in arms]
    check("E2: hybrid 3,200-5,000 tokens vs direct 16,000-31,000",
          3100 <= min(tok_h) and max(tok_h) <= 5000 and 16000 <= min(tok_d) and max(tok_d) <= 31000,
          f"{min(tok_h)}-{max(tok_h)} vs {min(tok_d)}-{max(tok_d)}")
    rat = [float(arms[m]["direct_usd_per_correct"]) / float(arms[m]["hybrid_usd_per_correct"]) for m in arms]
    check("E2: cost per correct schedule 4-16 times lower", 3.7 <= min(rat) and max(rat) <= 16.5,
          f"{min(rat):.1f}-{max(rat):.1f}")
    faster = sum(float(arms[m]["hybrid_latency_s"]) < float(arms[m]["direct_latency_s"]) for m in arms)
    check("E2: hybrid faster for four of five models", faster == 4, faster)
    dr = {r["model"]: r for r in rows("c2_drift.csv")}
    same = sum(int(r["oct_equals_july"]) for r in dr.values())
    within = sum(int(r["july_rep0_equals_rep12"]) for r in dr.values())
    pairs = sum(int(r["july_pairs"]) for r in dr.values())
    check("E2: drift 86% vs 88%", round(100 * same / 65) == 86 and round(100 * within / pairs) == 88)
    check("E2: drift correct counts GPT 10, Gemini 13, Llama 3 (unchanged), Claude -1, Qwen +2",
          all(int(dr[m]["oct_correct"]) - int(dr[m]["july_correct"]) == d
              for m, d in (("gpt", 0), ("gemini", 0), ("llama-3.3", 0), ("claude", -1), ("qwen-3", 2)))
          and int(dr["llama-3.3"]["oct_correct"]) == 3)

    # ---------------------------------------------------------------- Exp 3
    e3 = {r["regime"]: r for r in rows("b3_exp3.csv")}
    val = {g: float(e3[g]["price_only_milp"]) - float(e3[g]["weather_aware_milp"]) for g in e3}
    check("E3: forecast worth GBP 0.09-0.22/day to the MILP",
          near(min(val.values()), 0.09, 0.006) and near(max(val.values()), 0.215, 0.006),
          {g: round(v, 3) for g, v in val.items()})
    check("E3: MILP value 0.133 [0.063, 0.211]; agents -0.026 [-0.058, 0.000]",
          [round(float(iv["Exp3 weather value, MILP"][k]), 3) for k in ("estimate", "ci_low", "ci_high")] == [0.133, 0.063, 0.211]
          and [round(float(iv["Exp3 weather value, agents (matched)"][k]), 3) for k in ("estimate", "ci_low", "ci_high")] == [-0.026, -0.058, 0.0])
    a2 = {(r["regime"], r["arm"]): float(r["scr"]) for r in rows("a2_table6_extended.csv")}
    scr = [(round(a2[(g, "Weather-aware agent")], 2), round(a2[(g, "Weather-aware MILP (forecast)")], 2))
           for g in ("overcast", "mixed", "sunny")]
    check("E3: SCR agent vs MILP 0.49/0.59, 0.59/0.78, 0.61/0.78 (two decimals)",
          scr == [(0.49, 0.59), (0.59, 0.78), (0.61, 0.78)] or scr == [(0.49, 0.59), (0.59, 0.78), (0.61, 0.77)], scr)
    sm = {(r["experiment"], r["model"]): r for r in rows("a3_schedule_match.csv")}
    check("E3/E4: no agent run reproduces the weather-aware MILP schedule where it differs",
          all(int(r["weather_milp"]) == 0 for r in sm.values()))
    check("E3: GPT reproduces the price-only MILP in 33 of 45 runs; E4: 12 of 21",
          int(sm[("exp3 weather-aware", "gpt")]["price_only_milp"]) == 33
          and int(sm[("exp4a", "gpt")]["price_only_milp"]) == 12)
    nc = {(r["regime"], r["arm"]): r["non_commit"] for r in rows("a2_table6_extended.csv")}
    check("E3: Gemini non-commits 12 of 45 (3+5+4)",
          [nc[(g, "Weather-aware agent")] for g in ("overcast", "mixed", "sunny")] == ["3/45", "5/45", "4/45"])
    check("E3: hybrid captures 96/100/100%",
          [round(float(e3[g]["hybrid_weather_value_captured_pct"])) for g in ("overcast", "mixed", "sunny")] == [96, 100, 100])
    hv = [float(x) for x in __import__("re").findall(r"hybrid\s+\+?(-?[\d.]+)\s+95% CI \[\+?(-?[\d.]+), \+?(-?[\d.]+)\]", b3)[0]]
    check("E3: hybrid +0.139 [0.069, 0.216] vs price-only agents", hv == [0.139, 0.069, 0.216], hv)
    h3 = logs("exp3-hybrid")
    base = json.loads((config.DATA / "results" / "baselines.json").read_text(encoding="utf-8"))["per_day"]
    match = sum((r["score"] or {}).get("committed_starts") == base[r["day"]]["wfh_ev18"]["oracle"]["starts"] for r in h3)
    nonc = sum((r["score"] or {}).get("net_cost_realized") is None for r in h3)
    check("E3: hybrid identical to weather-aware MILP in 134/135 runs, no non-commits",
          (match, len(h3), nonc) == (134, 135, 0), (match, len(h3), nonc))
    d3 = [r for r in logs("exp3") if r["scenario"] == "weather_aware"]
    t_h = st.mean(r["tokens"]["input"] + r["tokens"]["output"] for r in h3)
    t_d = st.mean(r["tokens"]["input"] + r["tokens"]["output"] for r in d3)
    check("E3: about 4,000 vs 22,500 tokens per run (about a fifth)",
          near(t_h, 4000, 100) and near(t_d, 22500, 200) and near(t_h / t_d, 0.2, 0.03), f"{t_h:.0f} vs {t_d:.0f}")
    d = "2026-05-26"
    check("E3 Fig: sunny day GBP 5.68 (weather-aware MILP) vs 6.22 (price-only MILP)",
          near(base[d]["wfh_ev18"]["oracle"]["net_cost_realized"], 5.68, 0.005)
          and near(base[d]["wfh_ev18"]["price_only_milp"]["net_cost_realized"], 6.22, 0.005))
    f5 = {r["noise"]: r for r in rows("a2_fig5_matched.csv")}
    small = [float(f5[k]["delta_vs_unperturbed"]) for k in ("noise_10_m", "noise_10_p")]
    large = [float(f5[k]["delta_vs_unperturbed"]) for k in ("noise_25_m", "noise_25_p", "noise_50_m", "noise_50_p")]
    check("E3 probe: +/-10% no measurable effect; +/-25-50% GBP 0.06-0.08/day",
          max(abs(x) for x in small) < 0.01 and near(min(large), 0.06, 0.005) and near(max(large), 0.08, 0.005),
          (small, large))

    # ---------------------------------------------------------------- Exp 4
    t7 = {r["policy"]: r for r in rows("a3_table7.csv")}
    ag = [float(t7[p]["week_cost_gbp"]) for p in ("GPT-4o-mini agent", "Gemini 2.5 Flash agent", "Claude Sonnet 4.6 agent")]
    timer = float(t7["Off-peak timer"]["week_cost_gbp"])
    red = [100 * (timer - a) / timer for a in ag]
    check("E4: agents GBP 24.63-25.00, 51-52% below the timer (51.27)",
          near(min(ag), 24.63, 0.006) and near(max(ag), 25.00, 0.006) and near(timer, 51.27, 0.006)
          and 51 <= min(red) and max(red) < 52.5, [round(x, 1) for x in red])
    check("E4: eta 96.7-98.0; price-only MILP 98.1",
          [float(t7[p]["timer_to_oracle_share_pct"]) for p in ("Claude Sonnet 4.6 agent", "GPT-4o-mini agent", "Price-only MILP")] == [96.7, 98.0, 98.1])
    wv = float(t7["Price-only MILP"]["week_cost_gbp"]) - float(t7["Weather-aware MILP (oracle)"]["week_cost_gbp"])
    check("E4: forecast worth GBP 0.52 over the week; agents capture -2/-29/-74%",
          near(wv, 0.52, 0.006) and [round(float(t7[p]["weather_value_captured_pct"])) for p in
                                    ("GPT-4o-mini agent", "Gemini 2.5 Flash agent", "Claude Sonnet 4.6 agent")] == [-2, -29, -74])
    check("E4: immediate and greedy miss the deadline on all 7 days; Gemini non-commits 3/21",
          t7["Immediate start"]["ev_deadline_met_days"] == "0/7" and t7["Greedy cheapest-slot heuristic"]["ev_deadline_met_days"] == "0/7"
          and t7["Gemini 2.5 Flash agent"]["non_commit"] == "3/21")
    wk = rows("a6_weekly_runs.csv")
    exact = sum(r["as_requested"] == "True" for r in wk)
    extra = sum(int(r["unrequested"]) > 0 for r in wk)
    miss = [int(r["missing"]) for r in wk if int(r["missing"]) > 0]
    sub = [100 * (float(r["subset_cost_gbp"]) - float(r["milp_0800_gbp"])) / float(r["milp_0800_gbp"])
           for r in wk if r["subset_cost_gbp"]]
    da = sum(float(r["day_alloc_gbp"]) for r in wk if r["day_alloc_gbp"])
    ti = sum(float(r["timing_gbp"]) for r in wk if r["timing_gbp"])
    check("Weekly: 0/10 exact, 9 with unrequested cycles, 1 omitting three",
          (exact, extra, miss) == (0, 9, [3]), (exact, extra, miss))
    check("Weekly: requested-only plans 54-66% above the weekly MILP; 95% of gap is timing",
          round(min(sub)) == 54 and round(max(sub)) == 66 and round(100 * ti / (da + ti)) == 95,
          f"{min(sub):.1f}-{max(sub):.1f}, {100 * ti / (da + ti):.0f}%")

    # ---------------------------------------------------------------- totals
    n = sum(1 for f in ("exp1", "exp2", "exp2-hybrid", "exp3", "exp3-hybrid", "exp3-noise",
                        "exp4a", "exp4b", "exp2-drift") for _ in open(RUNS / f"{f}.jsonl", encoding="utf-8"))
    check("Total: 2,118 logged runs", n == 2118, n)
    a5s = (OUT / "a5_summary.txt").read_text(encoding="utf-8")
    check("Methods: repeat agreement 90% (Exp 1) and 82% (Exp 2)", "(90%)" in a5s and "(82%)" in a5s)

    print(f"\n{sum(results)}/{len(results)} claims verified")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
