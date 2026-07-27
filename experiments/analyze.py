"""
G11: aggregate run logs into the paper's tables and apply the champion rule.

    python -m experiments.analyze --exp exp1        # Table IV + regime + Pareto csv
    python -m experiments.analyze --exp exp2        # Table V
    python -m experiments.analyze --champion        # writes champion.json (rule)
    python -m experiments.analyze --exp exp3        # Table VI + sensitivity
    python -m experiments.analyze --exp exp4        # Table VII + projections

Reads data/runs/<exp>.jsonl and data/results/baselines.json; writes CSVs
(and .tex fragments) to data/results/. Refuses to aggregate rows whose
config_hash differs from the majority (guards against blending configs).
Lead metric is cost gap; optimality-rate and success-rate reported beside.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from experiments import archive, config

RESULTS = config.DATA / "results"


def _load(exp: str) -> List[Dict]:
    log = config.RUNS / f"{exp}.jsonl"
    if not log.exists():
        raise SystemExit(f"{log} missing -- run the experiment first")
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    # A retried run (e.g. after a transient DeepInfra 429) can appear more than
    # once: an earlier errored attempt and a later successful one. Keep the
    # last non-errored row per run_id; fall back to the last row if every
    # attempt errored (so genuine persistent failures remain visible).
    by_id: Dict[str, Dict] = {}
    for r in rows:
        rid = r.get("run_id")
        if rid is None:
            by_id[id(r)] = r                      # unkeyed rows: keep as-is
            continue
        prev = by_id.get(rid)
        if prev is None or not r.get("error"):
            by_id[rid] = r
    rows = list(by_id.values())
    hashes = Counter(r.get("config_hash") for r in rows)
    if len(hashes) > 1:
        # By default, guard against accidental blends by keeping the majority
        # hash only. For the documented open-model extension the closed and
        # open runs are executed under two config hashes that differ ONLY by
        # the added model entries + pricing (tariff, days, appliances,
        # prompts, and scoring are byte-identical). To aggregate them together
        # -- a deliberate, documented merge -- declare BOTH hashes explicitly
        # via ANALYZE_CONFIG_HASHES (comma-separated). Nothing is merged that
        # is not named on that allowlist.
        import os
        declared = os.getenv("ANALYZE_CONFIG_HASHES", "").strip()
        if declared:
            allow = {h.strip() for h in declared.split(",") if h.strip()}
            present = set(hashes)
            unknown = present - allow
            if unknown:
                raise SystemExit(
                    f"config_hash {sorted(unknown)} present but not in "
                    f"ANALYZE_CONFIG_HASHES={sorted(allow)}; refusing to "
                    f"silently drop or blend. Add them explicitly if intended.")
            print(f"[merge] aggregating declared config_hashes {sorted(present)}")
            rows = [r for r in rows if r.get("config_hash") in allow]
        else:
            keep = hashes.most_common(1)[0][0]
            print(f"[warn] mixed config_hash {dict(hashes)}; keeping {keep} "
                  f"only (set ANALYZE_CONFIG_HASHES to merge deliberately)")
            rows = [r for r in rows if r.get("config_hash") == keep]
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def _rate(flags):
    flags = [f for f in flags if f is not None]
    return round(sum(bool(f) for f in flags) / len(flags), 4) if flags else None


def _write_csv(name: str, rows: List[Dict], cols: List[str]):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {p}")


# ------------------------------------------------------------------ exp1
def analyze_exp1():
    rows = [r for r in _load("exp1") if r["scenario"] == "multi"]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["interface"])].append(r)
    out = []
    for (model, iface), rs in sorted(cells.items()):
        scores = [r["score"] for r in rs if r["score"]]
        # Mean cost gap is computed over SUCCESSFUL runs only. A run that
        # violates a hard constraint (e.g. EV finishing after its deadline) is
        # already counted against the model via success_rate; its cost is not
        # comparable to the constrained MILP optimum (it can even be "cheaper"
        # than the optimum by using forbidden slots, producing a negative gap).
        # Averaging such runs into the gap conflates constraint compliance with
        # window-selection quality. Restricting to successful runs makes the gap
        # mean exactly "of the valid schedules produced, how far from optimal",
        # applied identically to every model so the comparison stays fair.
        ok_gaps = [s["cost_gap"] for s in scores
                   if s.get("success") and s.get("cost_gap") is not None]
        # The gap distribution is fat-tailed: typical runs are near-optimal
        # (median ~ 0) but rare large misses occur, especially for open models
        # on negative-price days (e.g. one Qwen fc run at ~123% of optimum).
        # A mean is a poor summary of such a distribution -- a single outlier
        # dominates it and can even make a worse interface look better on the
        # mean. We therefore report the MEDIAN as the robust central tendency,
        # with mean and max beside it so the tail stays visible rather than
        # hidden. All three are over successful runs only.
        _med = (round(100 * st.median(ok_gaps), 3) if ok_gaps else None)
        _mx = (round(100 * max(ok_gaps), 3) if ok_gaps else None)
        out.append({
            "model": model, "interface": iface, "n": len(rs),
            "success_rate": _rate([s["success"] for s in scores]),
            "optimality_rate": _rate([s["optimal"] for s in scores]),
            "near_optimal_rate": _rate(
                [(s["cost_gap"] is not None and s["cost_gap"] <= 0.01)
                 for s in scores]),
            "median_cost_gap_pct": _med,
            "mean_cost_gap_pct": (round(100 * _mean(ok_gaps), 3)
                                  if ok_gaps else None),
            "max_cost_gap_pct": _mx,
            "n_success_for_gap": len(ok_gaps),
            "mean_iters": _mean([r["iterations"] for r in rs]),
            "mean_tokens": _mean([r["tokens"]["input"] + r["tokens"]["output"]
                                  for r in rs]),
            "mean_latency_s": _mean([r["latency_s"] for r in rs]),
            "cost_per_success_usd": _cost_per_success(rs),
        })
    _write_csv("exp1_main.csv", out, list(out[0].keys()))
    # per-regime cost gap (FC only)
    sel = archive.load_selection()
    regime = {}
    for name, days in sel["exp1_terciles"].items():
        regime.update({d: name for d in days})
    per = defaultdict(list)
    for r in rows:
        if r["interface"] != "fc" or not r["score"]:
            continue
        per[(r["model"], regime.get(r["day"], "?"))].append(
            r["score"]["cost_gap"])
    reg_rows = [{"model": m, "regime": g,
                 "mean_cost_gap_pct": round(100 * _mean(v), 3)}
                for (m, g), v in sorted(per.items())]
    _write_csv("exp1_by_regime.csv", reg_rows,
               ["model", "regime", "mean_cost_gap_pct"])
    for o in out:
        print(o)


def _cost_per_success(rs):
    tot = sum(r["est_cost_usd"] or 0 for r in rs)
    ns = sum(1 for r in rs if r["score"] and r["score"]["success"])
    return round(tot / ns, 5) if ns else None


# ------------------------------------------------------------------ exp2
def analyze_exp2():
    rows = _load("exp2")
    scen_family = {}
    scen_dir = config.ARCHIVE / "scenarios"
    for entry in json.loads((scen_dir / "manifest.json").read_text()):
        scen_family[entry["id"]] = entry["family"]
    out = []
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["prompt_version"], r["scenario"])].append(r)
    for (model, pv, scen), rs in sorted(cells.items()):
        scores = [r["score"] for r in rs if r["score"]]
        fam = scen_family.get(scen, "?")
        rec = {"model": model, "prompt": pv, "scenario": scen, "family": fam,
               "n": len(rs)}
        if fam == "infeasible":
            rec["infeasibility_reported_rate"] = _rate(
                [s["infeasibility_reported"] for s in scores])
            rec["fabrication_rate"] = _rate(
                [s["fabricated_schedule"] for s in scores])
            rec["success_rate"] = _rate([s["success"] for s in scores])
        else:
            rec["deadline_satisfied_rate"] = _rate(
                [s["deadline_satisfied"] for s in scores])
            rec["violation_rate"] = _rate(
                [bool(s["constraint_violations"]) for s in scores])
            rec["success_rate"] = _rate([s["success"] for s in scores])
            raw_gap = _mean([s["cost_gap"] for s in scores
                                if s.get("cost_gap") is not None])
            rec["mean_cost_gap_pct"] = (round(100 * raw_gap, 3)
                                        if raw_gap is not None else None)
        out.append(rec)
    cols = sorted({k for r in out for k in r})
    _write_csv("exp2_main.csv", out, cols)
    for o in out:
        print(o)


# ------------------------------------------------------------- champion
def apply_champion():
    e1 = [r for r in _load("exp1")
          if r["scenario"] == "multi" and r["interface"] == "fc" and r["score"]]
    e2 = _load("exp2")
    prim, tie = {}, {}
    # Champion selection stays over the three closed models only: the champion
    # drives Exp 3-noise and Exp 4b, which are not run for the open-weight
    # models, so they are out of scope for this rule by construction.
    CLOSED = ["gpt", "gemini", "claude"]
    for m in config.MODELS:
        if m == "mock" or m not in CLOSED:
            continue
        s1 = [r["score"] for r in e1 if r["model"] == m]
        if not s1:
            continue
        succ = _rate([s["success"] for s in s1])
        opt = _rate([s["optimal"] for s in s1])
        prim[m] = (succ or 0) * (opt or 0)
        guided = [r["score"] for r in e2
                  if r["model"] == m and r["prompt_version"].endswith("guided")
                  and r["score"]]
        dsr = _rate([s.get("deadline_satisfied") for s in guided])
        inf = _rate([s.get("infeasibility_reported") for s in guided
                     if s.get("scenario_infeasible")])
        tie[m] = (dsr or 0) + (inf or 0)
    if not prim:
        raise SystemExit("no scored exp1 multi/fc rows for real models -- "
                         "run exp1 and exp2 with gpt/gemini/claude first")
    ranked = sorted(prim, key=lambda m: (prim[m], tie.get(m, 0)), reverse=True)
    champ = ranked[0]
    out = {"champion": champ,
           "rule": "primary=success*optimality (exp1 multi fc); "
                   "tiebreak=guided DSR+infeasibility (exp2)",
           "primary": prim, "tiebreak": tie, "ranking": ranked}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "champion.json").write_text(json.dumps(out, indent=2))
    print(f"[champion] {champ}  (primary={prim}, tiebreak={tie})")


# ------------------------------------------------------------------ exp3
def analyze_exp3():
    rows = _load("exp3")
    sel = archive.load_selection()
    regime = {d: r for r, ds in sel["exp3_regimes"].items() for d in ds}
    cells = defaultdict(list)
    for r in rows:
        if not r["score"]:
            continue
        cells[(regime.get(r["day"], "?"), r["scenario"])].append(r["score"])
    out = []
    for (reg, arm), sc in sorted(cells.items()):
        out.append({"regime": reg, "objective": arm,
                    "mean_net_cost_realized": _mean(
                        [s["net_cost_realized"] for s in sc]),
                    "mean_scr": _mean([s["scr_realized"] for s in sc]),
                    "n": len(sc)})
    _write_csv("exp3_main.csv", out,
               ["regime", "objective", "mean_net_cost_realized",
                "mean_scr", "n"])
    for o in out:
        print(o)
    # noise sensitivity (if present)
    npath = config.RUNS / "exp3-noise.jsonl"
    if npath.exists():
        nrows = [json.loads(l) for l in npath.read_text().splitlines() if l.strip()]
        by_all = defaultdict(list)
        for r in nrows:
            val = (r["score"] or {}).get("net_cost_realized")
            by_all[r["scenario"]].append(val)
            nz_rows = []
        for k, v_all in sorted(by_all.items()):
            v_scored = [x for x in v_all if x is not None]
            nz_rows.append({
                "noise": k,
                "n_total": len(v_all),
                "n_scored": len(v_scored),
                "success_rate": round(len(v_scored)/len(v_all), 4) if v_all else None,
                "mean_net_cost_realized": _mean(v_scored),
            })
        _write_csv("exp3_noise.csv", nz_rows,
                   ["noise","n_total","n_scored","success_rate",
                    "mean_net_cost_realized"])


# ------------------------------------------------------------------ exp4
def _bootstrap_4week(daily_vals: List[float], iters=5000):
    if not daily_vals:
        return None, None, None
    rng = random.Random(42)
    means = []
    for _ in range(iters):
        sample = [rng.choice(daily_vals) for _ in daily_vals]
        means.append(28 * st.mean(sample))
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return round(28 * st.mean(daily_vals), 2), round(lo, 2), round(hi, 2)


def analyze_exp4():
    rows = _load("exp4a")
    base = json.loads((RESULTS / "baselines.json").read_text())["per_day"]
    # agent daily realized cost, averaged over passes, per (model, day)
    per = defaultdict(list)
    for r in rows:
        if r["score"]:
            per[(r["model"], r["day"])].append(r["score"]["net_cost_realized"])
    models = sorted({m for m, _ in per})
    week_days = sorted({d for _, d in per})
    out = []
    # reference policies over the same week (WFH scenario used in exp4)
    for policy in ["immediate", "off_peak_timer", "greedy_slot",
                   "price_only_milp", "oracle"]:
        daily = [base[d]["wfh_ev18"][policy]["net_cost_realized"]
                 for d in week_days]
        wk = round(sum(daily), 3)
        proj, lo, hi = _bootstrap_4week(daily)
        out.append({"policy": policy, "week_cost": wk,
                    "proj_4wk": proj, "ci_lo": lo, "ci_hi": hi})
    for m in models:
        daily = [_mean(per[(m, d)]) for d in week_days]
        wk = round(sum(daily), 3)
        proj, lo, hi = _bootstrap_4week(daily)
        out.append({"policy": f"agent_{m}", "week_cost": wk,
                    "proj_4wk": proj, "ci_lo": lo, "ci_hi": hi})
    # savings share vs oracle/timer
    oracle_wk = next(o["week_cost"] for o in out if o["policy"] == "oracle")
    timer_wk = next(o["week_cost"] for o in out if o["policy"] == "off_peak_timer")
    gap = timer_wk - oracle_wk
    for o in out:
        o["savings_share_pct"] = (round(100 * (timer_wk - o["week_cost"]) / gap, 1)
                                  if gap > 1e-9 else None)
    _write_csv("exp4_main.csv", out,
               ["policy", "week_cost", "proj_4wk", "ci_lo", "ci_hi",
                "savings_share_pct"])
    for o in out:
        print(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["exp1", "exp2", "exp3", "exp4"])
    ap.add_argument("--champion", action="store_true")
    args = ap.parse_args()
    if args.champion:
        apply_champion()
    elif args.exp == "exp1":
        analyze_exp1()
    elif args.exp == "exp2":
        analyze_exp2()
    elif args.exp == "exp3":
        analyze_exp3()
    elif args.exp == "exp4":
        analyze_exp4()
    else:
        ap.error("give --exp <e> or --champion")


if __name__ == "__main__":
    main()
