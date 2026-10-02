"""
Phase A, step A2 -- Experiment 3 re-analysis with MILP reference arms.

Why
---
Table 6 compares the two AGENT arms (price-only vs weather-aware) and finds a
regime-dependent effect: cheaper on overcast days, dearer on sunny days. The
paper attributes the sunny-day loss to negative midday prices. Reviewer 2
asked whether it comes from forecast error or from favouring self-consumption;
Reviewer 3 asked for MILP reference arms per regime.

This script adds three deterministic arms (no LLM):
  price-only MILP      decides with no PV (G = 0), scored on actual PV
  weather-aware MILP   decides on FORECAST PV, scored on actual PV
                       (= "oracle" in compute_baselines.py)
  perfect-foresight    decides on ACTUAL PV (lower bound)

It also fixes a comparability problem in the agent arms. Realized cost exists
only for runs that commit all three appliances, and Table 6 averages whatever
runs have one. When a model fails to commit on some days in one arm only, the
two arms are averaged over different days. The script therefore reports a
MATCHED comparison: per (model, day), average each arm over its committed
runs, keep only (model, day) pairs present in both arms, then average.

Figure 5 (forecast-error sweep) has a related problem: its zero point is the
pooled three-model weather-aware mean from Table 6, while the perturbed points
are champion-only (GPT-4o-mini). The script rebuilds the sweep against GPT's
own unperturbed runs, matched by day.

No LLM calls. Run from the repo root:
    python -m revision.a2_exp3_milp_reference
    python plot_fig4_sensitivity.py --noise-csv revision/outputs/a2_fig5_matched.csv \
        --out revision/outputs/fig5_sensitivity_matched --legend-loc below

Outputs (revision/outputs/):
    a2_table6_extended.csv  regime x arm: matched net cost, SCR, n
    a2_per_model.csv        matched agent arms by model, with non-commit counts
    a2_table6.tex           LaTeX rows for the revised Table 6
    a2_fig5_matched.csv     Fig 5 input, champion-only and matched by day
    a2_summary.txt          console summary
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import statistics as st

from experiments import archive, config
from experiments.compute_baselines import standard_tasks, EV_WFH_LATEST_FINISH
from experiments.optimizer import solve, evaluate_schedule

ROOT = config.ROOT
RUNS = ROOT / "data" / "runs" / "exp3.jsonl"
BASELINES = ROOT / "data" / "results" / "baselines.json"
EXP3_MAIN = ROOT / "data" / "results" / "exp3_main.csv"
NOISE_RUNS = ROOT / "data" / "runs" / "exp3-noise.jsonl"
NOISE_MAIN = ROOT / "data" / "results" / "exp3_noise.csv"
CHAMPION = json.loads((ROOT / "data" / "results" / "champion.json").read_text(encoding="utf-8"))["champion"]
OUT = ROOT / "revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

F = config.EXPORT_RATE_GBP
REGIMES = ["overcast", "mixed", "sunny"]
ARMS = ["price_only", "weather_aware"]
ARM_LABEL = {"price_only": "Price-only agent", "weather_aware": "Weather-aware agent"}
MILP_LABEL = {"po_milp": "Price-only MILP", "wa_milp": "Weather-aware MILP (forecast)",
              "pf_milp": "Perfect-foresight MILP"}
MODELS = ["gpt", "gemini", "claude"]
TOL = 1e-3


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def milp_arms(day: str) -> dict:
    prices = list(archive.load_prices(day))
    pv_fc, pv_act = archive.pv_slots(day, "fc"), archive.pv_slots(day, "actual")
    tasks = standard_tasks(EV_WFH_LATEST_FINISH)
    plans = {"po_milp": solve(prices, tasks),
             "wa_milp": solve(prices, tasks, pv=pv_fc, export_rate=F),
             "pf_milp": solve(prices, tasks, pv=pv_act, export_rate=F)}
    out = {"min_price": min(prices), "neg_slots": sum(p < 0 for p in prices)}
    for arm, res in plans.items():
        assert res["status"] == "optimal", f"{day} {arm}: {res['status']}"
        ev = evaluate_schedule(res["starts"], tasks, prices, export_rate=F, pv=pv_act)
        out[arm] = {"cost": ev["net_cost"], "scr": ev["self_consumption_ratio"]}
    return out


def main():
    sel = archive.load_selection()
    days_of = sel["exp3_regimes"]
    regime_of = {d: r for r, ds in days_of.items() for d in ds}
    rows = [json.loads(l) for l in open(RUNS, encoding="utf-8")]
    base = json.loads(BASELINES.read_text(encoding="utf-8"))["per_day"]
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("A2 -- Exp 3 re-analysis with MILP reference arms (realized net cost, GBP/day)")
    say("")

    # ---------------------------------------------------------------- MILP
    milp = {d: milp_arms(d) for d in regime_of}
    for d in milp:
        for arm, key in (("po_milp", "price_only_milp"), ("wa_milp", "oracle")):
            ref = base[d]["wfh_ev18"][key]["net_cost_realized"]
            assert abs(milp[d][arm]["cost"] - ref) < TOL, f"{d} {arm} != baselines.json"
    say(f"Sanity check 1 -- MILP arms match baselines.json on all {len(milp)} days")

    # ---------------------------------------------------------------- agents
    runs = collections.defaultdict(list)          # (model, day, arm) -> [(cost, scr)]
    attempts = collections.Counter()               # (regime, arm, model) -> runs
    noncommit = collections.Counter()              # ... -> runs without realized cost
    unbalanced = collections.defaultdict(list)     # (regime, arm) -> costs (as Table 6)
    for r in rows:
        s = r.get("score")
        if not s:
            continue
        reg = regime_of[r["day"]]
        attempts[(reg, r["scenario"], r["model"])] += 1
        unbalanced[(reg, r["scenario"])].append(s["net_cost_realized"])
        if s["net_cost_realized"] is None:
            noncommit[(reg, r["scenario"], r["model"])] += 1
        else:
            runs[(r["model"], r["day"], r["scenario"])].append(
                (s["net_cost_realized"], s["scr_realized"]))

    published = {(x["regime"], x["objective"]): float(x["mean_net_cost_realized"])
                 for x in csv.DictReader(open(EXP3_MAIN))}
    for k, v in unbalanced.items():
        assert abs(mean(v) - published[k]) < TOL, f"{k} does not reproduce exp3_main.csv"
    say("Sanity check 2 -- unbalanced agent means reproduce published Table 6")
    say("")

    say("Runs with no realized cost (did not commit all three appliances):")
    any_nc = False
    for reg in REGIMES:
        for arm in ARMS:
            for m in MODELS:
                if noncommit[(reg, arm, m)]:
                    any_nc = True
                    say(f"  {reg:9s} {ARM_LABEL[arm]:20s} {m:7s} "
                        f"{noncommit[(reg, arm, m)]}/{attempts[(reg, arm, m)]}")
    if not any_nc:
        say("  none")
    say("")

    def matched(reg, models):
        pairs = [(m, d) for m in models for d in days_of[reg]
                 if runs[(m, d, "price_only")] and runs[(m, d, "weather_aware")]]
        res = {"pairs": len(pairs)}
        for arm in ARMS:
            res[arm] = {
                "cost": mean([mean([c for c, _ in runs[(m, d, arm)]]) for m, d in pairs]),
                "scr": mean([mean([s for _, s in runs[(m, d, arm)]]) for m, d in pairs])}
        return res

    # ---------------------------------------------------------------- table
    table, T = [], {}
    for reg in REGIMES:
        mt = matched(reg, MODELS)
        for arm in ARMS:
            nc = sum(noncommit[(reg, arm, m)] for m in MODELS)
            at = sum(attempts[(reg, arm, m)] for m in MODELS)
            row = {"regime": reg, "arm": ARM_LABEL[arm], "net_cost": mt[arm]["cost"],
                   "scr": mt[arm]["scr"], "n": f"{mt['pairs']} model-days",
                   "non_commit": f"{nc}/{at}",
                   "published_unbalanced": round(published[(reg, arm)], 4)}
            table.append(row); T[(reg, row["arm"])] = row
        for arm, label in MILP_LABEL.items():
            row = {"regime": reg, "arm": label,
                   "net_cost": mean([milp[d][arm]["cost"] for d in days_of[reg]]),
                   "scr": mean([milp[d][arm]["scr"] for d in days_of[reg]]),
                   "n": f"{len(days_of[reg])} days", "non_commit": "",
                   "published_unbalanced": ""}
            table.append(row); T[(reg, label)] = row

    say("Matched comparison (agents averaged over the same model-days in both arms):")
    say(f"  {'regime':9s} {'arm':31s} {'net cost':>9s} {'SCR':>6s}  {'n':>13s}  "
        f"{'non-commit':>10s}  {'Table 6 as published':>20s}")
    for t in table:
        say(f"  {t['regime']:9s} {t['arm']:31s} {t['net_cost']:9.3f} {t['scr']:6.3f}  "
            f"{t['n']:>13s}  {t['non_commit']:>10s}  {str(t['published_unbalanced']):>20s}")
    say("")

    g = lambda reg, a: T[(reg, a)]["net_cost"]
    say("Value of weather information, GBP/day (positive = weather arm cheaper):")
    say(f"  {'regime':9s} {'MILP forecast':>14s} {'MILP perfect':>13s} "
        f"{'agents matched':>15s} {'Table 6 published':>18s}")
    for reg in REGIMES:
        say(f"  {reg:9s} "
            f"{g(reg, 'Price-only MILP') - g(reg, 'Weather-aware MILP (forecast)'):14.3f} "
            f"{g(reg, 'Price-only MILP') - g(reg, 'Perfect-foresight MILP'):13.3f} "
            f"{g(reg, 'Price-only agent') - g(reg, 'Weather-aware agent'):15.3f} "
            f"{published[(reg, 'price_only')] - published[(reg, 'weather_aware')]:18.3f}")
    say("")

    say("Self-consumption: weather-aware agent vs weather-aware MILP (matched agents):")
    for reg in REGIMES:
        a, m = T[(reg, "Weather-aware agent")], T[(reg, "Weather-aware MILP (forecast)")]
        say(f"  {reg:9s} agent SCR {a['scr']:.3f} vs MILP SCR {m['scr']:.3f}; "
            f"agent cost {a['net_cost'] - m['net_cost']:+.3f} GBP/day vs MILP")
    say("")

    say("Negative import prices (tests the current Sec. 5.4 explanation):")
    for reg in REGIMES:
        ds = days_of[reg]
        say(f"  {reg:9s} days with any negative slot {sum(milp[d]['neg_slots'] > 0 for d in ds)}"
            f"/{len(ds)}, lowest price {min(milp[d]['min_price'] for d in ds):.3f} GBP/kWh")
    say("")

    say("By model, matched (weather-aware minus price-only, GBP/day; SCR change):")
    pm_rows = []
    for reg in REGIMES:
        parts = []
        for m in MODELS:
            mt = matched(reg, [m])
            d_cost = mt["weather_aware"]["cost"] - mt["price_only"]["cost"]
            d_scr = mt["weather_aware"]["scr"] - mt["price_only"]["scr"]
            parts.append(f"{m} {d_cost:+.3f} ({d_scr:+.3f})")
            for arm in ARMS:
                pm_rows.append({"regime": reg, "model": m, "arm": ARM_LABEL[arm],
                                "net_cost": round(mt[arm]["cost"], 4),
                                "scr": round(mt[arm]["scr"], 4),
                                "matched_days": mt["pairs"],
                                "non_commit": f"{noncommit[(reg, arm, m)]}/{attempts[(reg, arm, m)]}"})
        say(f"  {reg:9s} " + "   ".join(parts))


    # ---------------------------------------------------------------- Fig 5
    say("")
    say(f"Figure 5 -- forecast-error sweep (champion: {CHAMPION}):")
    nruns = [json.loads(l) for l in open(NOISE_RUNS, encoding="utf-8")]
    pub_noise = {x["noise"]: float(x["mean_net_cost_realized"])
                 for x in csv.DictReader(open(NOISE_MAIN))}
    by_level = collections.defaultdict(list)
    for r in nruns:
        by_level[r["scenario"]].append((r["day"], (r["score"] or {}).get("net_cost_realized")))
    # Each run_id is a hash of its noise tag (grids.exp3_noise_specs), so the
    # label on every logged run can be verified independently of any CSV.
    tags = [f"noise_{int(L * 100)}_{s_}" for L in config.NOISE_LEVELS for s_ in "pm"]
    def rid(**kw):
        return hashlib.sha1(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:12]
    bad = 0
    for r in nruns:
        hit = [t for t in tags for rep in range(config.REPS_EXP3)
               if rid(exp="exp3-noise", scenario=t, model=CHAMPION, interface="fc",
                      day=r["day"], rep=rep, pv=r["prompt_version"]) == r["run_id"]]
        bad += hit != [r["scenario"]]
    assert bad == 0, f"{bad} noise runs whose label does not match their run_id"
    say(f"  Sanity check 3 -- all {len(nruns)} noise-run labels verified against their run_ids")
    mism = [lvl for lvl, v in sorted(by_level.items())
            if abs(mean([c for _, c in v]) - pub_noise[lvl]) >= TOL]
    if mism:
        say("  WARNING -- data/results/exp3_noise.csv disagrees with the run logs for: "
            + ", ".join(f"{l} (csv {pub_noise[l]:.3f}, logs {mean([c for _, c in by_level[l]]):.3f})"
                        for l in mism))
        say("  The published Figure 5 was drawn from that CSV; the values below use the logs.")
    else:
        say("  exp3_noise.csv matches the run logs")
    all_days = list(regime_of)
    base_day = {d: mean([c for c, _ in runs[(CHAMPION, d, "weather_aware")]]) for d in all_days}
    po_day = {d: mean([c for c, _ in runs[(CHAMPION, d, "price_only")]]) for d in all_days}
    base_all = mean([base_day[d] for d in all_days if base_day[d] is not None])
    po_all = mean([po_day[d] for d in all_days if po_day[d] is not None])
    pooled_zero = mean([published[(reg, "weather_aware")] for reg in REGIMES])
    say(f"  Zero point used in the published figure (3-model pooled mean): {pooled_zero:.3f}")
    say(f"  {CHAMPION} unperturbed weather-aware mean (15 days):            {base_all:.3f}")
    say(f"  {CHAMPION} price-only mean (15 days):                           {po_all:.3f}")
    say(f"  {'error':>6s} {'logs, unmatched':>16s} {'matched delta':>14s} {'plotted':>8s} {'days':>5s}")
    fig5_rows = [{"noise": "accurate", "mean_net_cost_realized": round(base_all, 4),
                  "delta_vs_unperturbed": 0.0, "n_days": len(all_days)},
                 {"noise": "price_only", "mean_net_cost_realized": round(po_all, 4),
                  "delta_vs_unperturbed": "", "n_days": len(all_days)}]
    order = ["noise_50_m", "noise_25_m", "noise_10_m", "noise_10_p", "noise_25_p", "noise_50_p"]
    for lvl in order:
        per_day = collections.defaultdict(list)
        for d, c in by_level[lvl]:
            if c is not None:
                per_day[d].append(c)
        days = [d for d in per_day if base_day.get(d) is not None]
        delta = mean([mean(per_day[d]) - base_day[d] for d in days])
        plotted = base_all + delta
        sign = "-" if lvl.endswith("_m") else "+"
        say(f"  {sign}{lvl.split('_')[1]:>4s}% {mean([c for _, c in by_level[lvl]]):16.3f} {delta:+14.3f} "
            f"{plotted:8.3f} {len(days):5d}")
        fig5_rows.append({"noise": lvl, "mean_net_cost_realized": round(plotted, 4),
                          "delta_vs_unperturbed": round(delta, 4), "n_days": len(days)})
    say("  plotted = unperturbed mean + matched delta (each level vs the same days, unperturbed)")
    with open(OUT / "a2_fig5_matched.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fig5_rows[0].keys()))
        w.writeheader(); w.writerows(fig5_rows)

    # ---------------------------------------------------------------- write
    with open(OUT / "a2_table6_extended.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for t in table:
            w.writerow({**t, "net_cost": round(t["net_cost"], 4), "scr": round(t["scr"], 4)})
    with open(OUT / "a2_per_model.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pm_rows[0].keys()))
        w.writeheader(); w.writerows(pm_rows)

    cols = list(ARM_LABEL.values()) + list(MILP_LABEL.values())
    tex = ["% Revised Table 6 body: realized net cost (GBP/day), SCR in parentheses.",
           "% Agent arms matched by model-day; MILP arms averaged over the regime's days.",
           "% Columns: " + " | ".join(cols)]
    for reg in REGIMES:
        tex.append(f"{reg.capitalize()} & " + " & ".join(
            f"{T[(reg, c)]['net_cost']:.3f} ({T[(reg, c)]['scr']:.3f})" for c in cols) + r" \\")
    (OUT / "a2_table6.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (OUT / "a2_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote revision/outputs/a2_table6_extended.csv, a2_per_model.csv, "
          "a2_table6.tex, a2_fig5_matched.csv, a2_summary.txt")


if __name__ == "__main__":
    main()
