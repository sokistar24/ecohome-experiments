"""
Phase B, step B3 -- Direct vs guarded vs hybrid scheduling.

Arms (same models, scenarios, repeats; all scored with the A1 deadline rule):
  direct   the original Exp 2 runs with the guided prompt (v2-guided): the LLM
           reads prices, chooses slots and commits them.
  guard    the direct runs passed through a deterministic validator. Each
           committed schedule is checked against the constraints the SAME model
           extracted in structured form for the same scenario and repeat (from
           its hybrid run). A violating commit, or an incomplete one without an
           infeasibility report, is replaced by the optimizer's schedule under
           those constraints; if they are infeasible the commit is blocked and
           infeasibility reported. A run that committed nothing passes through,
           and an agent's own infeasibility report is always kept.
  hybrid   the LLM only extracts requirements; optimize_schedule solves the MILP
           and commits (revision/hybrid.py, prompt v3.1-hybrid).

Because the optimizer never violates the constraints it is given, any hybrid
run that violates a stated constraint must have passed a wrong constraint: an
interpretation error, not an action error. The script classifies every hybrid
failure accordingly.

Exp 3: the hybrid arm against the direct agents and the MILP references from A2.

No LLM calls. Run from the repo root (after Phase A and B2):
    python -m revision.b3_hybrid_analysis
    python plot_fig3_taxonomy.py --csv revision/outputs/b3_fig_direct_vs_hybrid.csv \\
        --left-prompt direct --right-prompt hybrid \\
        --left-title "(a) Direct scheduling" --right-title "(b) Hybrid (LLM to MILP)" \\
        --out revision/outputs/fig_direct_vs_hybrid

Outputs (revision/outputs/): b3_exp2_arms.csv/.tex, b3_exp2_by_family.csv,
b3_hybrid_failures.csv, b3_extraction.csv, b3_exp3.csv/.tex,
b3_fig_direct_vs_hybrid.csv, b3_summary.txt
"""
from __future__ import annotations

import collections
import copy
import csv
import json

from experiments import archive, config
from experiments.eval_tools import RUN, _resolve_date
from experiments.optimizer import ApplianceTask, evaluate_schedule, solve
from experiments.runner import run_id
from revision.a1_rescore_departure_buffer import (
    load_scenarios, detect_buffered, rescore, correct, family, FAMKEY, FAMILIES)
from revision.a2_exp3_milp_reference import milp_arms
from revision.a4_exp2_violation_severity import measure, starts_of
from revision.a5_metrics_and_intervals import wilson, boot, mean
from revision.hybrid import PROMPT_VERSION_HYBRID, _to_slot

ROOT = config.ROOT
RUNS = config.RUNS
OUT = ROOT / "revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODELS5 = ["gpt", "gemini", "claude", "llama-3.3", "qwen-3"]
MODELS3 = ["gpt", "gemini", "claude"]
ARMS = ["direct", "guard", "hybrid"]
APP = config.APPLIANCES
ALL_THREE = ["washing_machine", "dishwasher", "ev_charger"]


def load(name):
    return [json.loads(l) for l in open(RUNS / f"{name}.jsonl", encoding="utf-8")]


def rep_of(row, exp, pv):
    for rep in range(3):
        if run_id(exp=exp, scenario=row["scenario"], model=row["model"], interface="fc",
                  day=row["day"], rep=rep, pv=pv) == row["run_id"]:
            return rep
    raise ValueError(f"cannot recover repeat for {row['run_id']}")


def replay_registry(row, inject_failure):
    """Rebuild the hybrid tool's final requirements registry from the logged
    optimize_schedule calls, with the tool's own semantics (hybrid.py)."""
    RUN.reset(row["eval_date"])
    date, reg, cap, n_valid = None, {}, None, 0
    calls = [c["args"] for c in row["tool_calls"] if c["name"] == "optimize_schedule"]
    if inject_failure and calls:
        calls = calls[1:]                  # the first executed call returned the 503
    for a in calls:
        req = [x for x in ALL_THREE if a.get(x)]
        if not req:
            continue
        try:
            d = _resolve_date(a.get("date", "tomorrow"))
            archive.load_prices(d)
            upd = {x: {"not_before": _to_slot(a.get(f"{x}_not_before"), finish=False),
                       "finish_by": _to_slot(a.get(f"{x}_finish_by"), finish=True)}
                   for x in req}
        except Exception:
            continue
        if date != d:
            date, reg, cap = d, {}, None
        reg.update(upd)
        if a.get("power_cap_kw") is not None:
            cap = a["power_cap_kw"]
        n_valid += 1
    return {"date": date, "reg": reg, "cap": cap, "n_calls": len(calls), "n_valid": n_valid}


def solve_registry(day, reg, cap):
    tasks = [ApplianceTask(x, APP[x]["power_kw"], APP[x]["slots"],
                           earliest_start=c["not_before"] or 0, latest_finish=c["finish_by"])
             for x, c in sorted(reg.items())]
    res = solve(list(archive.load_prices(day)), tasks, power_cap_kw=cap)
    return res["starts"] if res["status"] == "optimal" else None


def violates(starts: dict, reg: dict, cap) -> bool:
    for x, s in starts.items():
        c = reg.get(x)
        if c is None:
            continue
        if c["not_before"] is not None and s < c["not_before"]:
            return True
        if c["finish_by"] is not None and s + APP[x]["slots"] > c["finish_by"]:
            return True
    if cap is not None:
        load = [0.0] * config.SLOTS_PER_DAY
        for x, s in starts.items():
            for t in range(s, min(len(load), s + APP[x]["slots"])):
                load[t] += APP[x]["power_kw"]
        if max(load) > cap + 1e-9:
            return True
    return False


def main():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    scen = load_scenarios()
    buffered = detect_buffered(scen)

    def truth(sid):
        c = copy.deepcopy(scen[sid]["constraints"])
        if sid in buffered:
            c["ev_latest_finish"] = buffered[sid]
        return c

    direct = [r for r in load("exp2") if r["prompt_version"] == "v2-guided"]
    hybrid = load("exp2-hybrid")
    assert all(r["prompt_version"] == PROMPT_VERSION_HYBRID for r in hybrid)
    say(f"B3 -- direct vs guard vs hybrid. Exp 2: {len(direct)} direct-guided runs, "
        f"{len(hybrid)} hybrid runs")

    H = {(r["model"], r["scenario"], rep_of(r, "exp2-hybrid", PROMPT_VERSION_HYBRID)): r
         for r in hybrid}
    D = {(r["model"], r["scenario"], rep_of(r, "exp2", "v2-guided")): r for r in direct}
    assert set(H) == set(D), "direct and hybrid grids do not pair up"
    say(f"Paired by model x scenario x repeat: {len(H)} pairs")

    # ---------------------------------------------------------- replay check
    reg_of, mismatch, ties = {}, 0, 0
    for k, r in H.items():
        inj = scen[r["scenario"]]["constraints"].get("inject_price_tool_failure", False)
        rg = replay_registry(r, inj)
        reg_of[k] = rg
        if rg["n_valid"]:
            # solve on the date the model passed (it may differ from the scenario day)
            got = solve_registry(rg["date"], rg["reg"], rg["cap"]) or {}
            logged = {x: v["slot"] for x, v in (r["committed"] or {}).items()}
            if got != logged:
                # equal-cost alternative optimum (CBC tie-breaking can differ by platform)
                if set(got) == set(logged) and got:
                    tasks = [ApplianceTask(x, APP[x]["power_kw"], APP[x]["slots"]) for x in got]
                    pr = list(archive.load_prices(rg["date"]))
                    c1 = evaluate_schedule(got, tasks, pr)["net_cost"]
                    c2 = evaluate_schedule(logged, tasks, pr)["net_cost"]
                    if abs(c1 - c2) < 1e-9 and not violates(logged, rg["reg"], rg["cap"]):
                        ties += 1
                        continue
                mismatch += 1
    say(f"Sanity check -- replayed optimizer inputs reproduce the logged commitment in "
        f"{len(H) - mismatch}/{len(H)} hybrid runs"
        + (f" ({ties} via an equal-cost alternative optimum)" if ties else ""))
    if mismatch:
        raise SystemExit("Replay does not reproduce the logged hybrid commitments; stopping.")
    say("")

    # ---------------------------------------------------------- score the arms
    out = {}
    for k in sorted(H):
        m, sid, rep = k
        cons = truth(sid)
        d_row, h_row = D[k], H[k]
        d_sc = rescore(d_row, scen, cons)
        h_sc = rescore(h_row, scen, cons)
        # guard
        rg = reg_of[k]
        d_starts = starts_of(d_row)
        g_row = {"scenario": sid, "committed": d_row["committed"],
                 "infeasibility_report": d_row["infeasibility_report"]}
        action = "pass"
        if rg["n_valid"] and d_starts:
            day = scen[sid]["day"]
            fix = solve_registry(day, rg["reg"], rg["cap"])
            if fix is None:
                g_row = {"scenario": sid, "committed": {},
                         "infeasibility_report": "guard: extracted constraints infeasible"}
                action = "block"
            elif violates(d_starts, rg["reg"], rg["cap"]) or (
                    set(d_starts) != set(ALL_THREE) and not d_row["infeasibility_report"]):
                # an incomplete commit is only repaired when the agent did not
                # report infeasibility; its report, if any, is kept
                merged = {x: {"slot": s, "date": day} for x, s in d_starts.items()}
                merged.update({x: {"slot": s, "date": day} for x, s in fix.items()})
                g_row = {"scenario": sid, "committed": merged,
                         "infeasibility_report": d_row["infeasibility_report"]}
                action = "repair"
        g_sc = rescore(g_row, scen, cons)
        out[k] = {"direct": correct(sid, d_sc), "guard": correct(sid, g_sc),
                  "hybrid": correct(sid, h_sc), "guard_action": action,
                  "d_row": d_row, "h_row": h_row, "h_sc": h_sc, "d_sc": d_sc}

    # ---------------------------------------------------------- headline table
    say("Exp 2 correct-behaviour rate (A1 rule), 39 runs per cell; 95% Wilson interval with")
    say("unit = scenario (13):")
    say(f"  {'model':10s}" + "".join(f"{a:>22s}" for a in ARMS) + f"{'hybrid - direct':>24s}")
    arm_rows = []
    for m in MODELS5:
        cells = []
        for a in ARMS:
            units = [mean([out[(m, sid, rep)][a] for rep in range(3)])
                     for sid in sorted({k[1] for k in out})]
            est = mean(units)
            lo, hi = wilson(est, len(units))
            cells.append((est, lo, hi))
        diffs = [mean([out[(m, sid, rep)]["hybrid"] - out[(m, sid, rep)]["direct"]
                       for rep in range(3)]) for sid in sorted({k[1] for k in out})]
        dlo, dhi = boot(diffs, lambda u: mean(u))
        say(f"  {m:10s}" + "".join(f"{100*e:6.0f}% [{100*lo:3.0f},{100*hi:4.0f}]".rjust(22)
                                   for e, lo, hi in cells)
            + f"{100*mean(diffs):+7.0f} pp [{100*dlo:+.0f},{100*dhi:+.0f}]".rjust(24))
        arm_rows.append({"model": m, **{f"{a}_rate": round(cells[i][0], 3) for i, a in enumerate(ARMS)},
                         **{f"{a}_ci": f"[{100*cells[i][1]:.0f}, {100*cells[i][2]:.0f}]"
                            for i, a in enumerate(ARMS)},
                         "hybrid_minus_direct_pp": round(100 * mean(diffs), 1),
                         "diff_ci": f"[{100*dlo:+.0f}, {100*dhi:+.0f}]"})
    say("")

    # ---------------------------------------------------------- by family
    say("By scenario family, correct rate (direct / guard / hybrid):")
    say(f"  {'model':10s}" + "".join(f"{f:>16s}" for f in FAMILIES))
    fam_rows = []
    for m in MODELS5:
        parts = []
        for f in FAMILIES:
            ks = [k for k in out if k[0] == m and family(k[1]) == f]
            v = [mean([out[k][a] for k in ks]) for a in ARMS]
            parts.append("/".join(f"{x:.2f}" for x in v))
            fam_rows.append({"model": m, "family": f, **{a: round(v[i], 3) for i, a in enumerate(ARMS)}})
        say(f"  {m:10s}" + "".join(f"{p:>16s}" for p in parts))
    say("")

    say("Paired outcomes, direct vs hybrid (same model, scenario, repeat):")
    for m in MODELS5:
        ks = [k for k in out if k[0] == m]
        c = collections.Counter((out[k]["direct"], out[k]["hybrid"]) for k in ks)
        say(f"  {m:10s} both correct {c[(True, True)]:2d}  fixed by hybrid {c[(False, True)]:2d}  "
            f"broken by hybrid {c[(True, False)]:2d}  both wrong {c[(False, False)]:2d}")
    say("")

    # ---------------------------------------------------------- invalid commits
    say("Invalid commits on feasible scenarios (a committed schedule breaking a stated constraint):")
    for m in MODELS5:
        parts = []
        for a, key in (("direct", "d_row"), ("hybrid", "h_row")):
            inv, worst = 0, 0
            for k, o in out.items():
                if k[0] != m or family(k[1]) == "S4":
                    continue
                sev = measure(starts_of(o[key]), truth(k[1]))
                if any(v > 0 for v in sev.values()):
                    inv += 1
                    worst = max(worst, sev["deadline_overrun_min"])
            parts.append(f"{a} {inv:2d} (worst {worst / 60:.1f} h late)")
        say(f"  {m:10s} " + "   ".join(parts))
    say("")

    # ---------------------------------------------------------- hybrid failure taxonomy
    say("Why hybrid runs fail (each failed hybrid run, first matching cause):")
    tax_rows = []
    for k, o in sorted(out.items()):
        if o["hybrid"]:
            continue
        m, sid, rep = k
        cons, rg, h = truth(sid), reg_of[k], o["h_row"]
        st_ = starts_of(h)
        reg, cap = rg["reg"], rg["cap"]
        lf, ne, tcap = cons.get("ev_latest_finish"), cons.get("noisy_earliest_start"), cons.get("power_cap_kw")
        if rg["n_valid"] == 0:
            cause = ("no optimizer call: reported infeasible without checking"
                     if h["infeasibility_report"] else "no optimizer call")
        elif family(sid) == "S4":
            cause = "committed a schedule for an infeasible request"
        elif any(v > 0 for v in measure(st_, cons).values()):
            ev = reg.get("ev_charger", {})
            if lf is not None and (ev.get("finish_by") is None or ev["finish_by"] > lf):
                cause = "interpretation: missed or loosened the EV deadline"
            elif tcap is not None and (cap is None or cap > tcap):
                cause = "interpretation: missed the power cap"
            else:
                cause = "interpretation: missed quiet hours"
        elif not st_ and h["infeasibility_report"]:
            spurious = []
            if cap is not None and (tcap is None or cap < tcap):
                spurious.append("power cap")
            for x, c in reg.items():
                if x != "ev_charger" and c["finish_by"] is not None:
                    spurious.append(f"{x} deadline")
                if c["not_before"] and not (ne and x != "ev_charger"):
                    spurious.append(f"{x} earliest start")
            ev = reg.get("ev_charger", {})
            if lf is not None and ev.get("finish_by") is not None and ev["finish_by"] < lf:
                spurious.append("earlier EV deadline")
            cause = ("interpretation: added constraint(s) -> false infeasibility ("
                     + ", ".join(sorted(set(spurious)) or ["unclear"]) + ")")
        elif set(st_) != set(ALL_THREE):
            cause = ("tool failure not retried" if family(sid) == "S6"
                     else "incomplete: not every appliance scheduled")
        else:
            cause = "other"
        tax_rows.append({"model": m, "scenario": sid, "rep": rep, "cause": cause})
    c = collections.Counter((t["cause"]) for t in tax_rows)
    for cause, n in c.most_common():
        who = collections.Counter(t["model"] for t in tax_rows if t["cause"] == cause)
        say(f"  {n:3d}  {cause}  [{', '.join(f'{m} {v}' for m, v in sorted(who.items()))}]")
    say(f"  hybrid failures in total: {len(tax_rows)}")
    say("")

    # ---------------------------------------------------------- extraction accuracy
    say("Extraction accuracy on the binding constraints (hybrid runs that called the optimizer):")
    ext_rows = []
    for m in MODELS5:
        cnt = collections.Counter()
        for k, rg in reg_of.items():
            if k[0] != m or not rg["n_valid"]:
                continue
            cons = truth(k[1])
            reg = rg["reg"]
            if cons.get("ev_latest_finish") is not None:
                fb = reg.get("ev_charger", {}).get("finish_by")
                cnt["ev_n"] += 1
                cnt["ev_ok"] += fb == cons["ev_latest_finish"]
            if cons.get("power_cap_kw") is not None:
                cnt["cap_n"] += 1
                cnt["cap_ok"] += rg["cap"] == cons["power_cap_kw"]
            else:
                cnt["nocap_n"] += 1
                cnt["cap_spurious"] += rg["cap"] is not None
            if cons.get("noisy_earliest_start") is not None:
                cnt["quiet_n"] += 1
                cnt["quiet_ok"] += all(reg.get(x, {}).get("not_before") == cons["noisy_earliest_start"]
                                       for x in ("washing_machine", "dishwasher"))
            cnt["cover_n"] += 1
            cnt["cover_ok"] += set(reg) == set(ALL_THREE)
        spur_vals = sorted({rg["cap"] for k, rg in reg_of.items() if k[0] == m and rg["n_valid"]
                            and truth(k[1]).get("power_cap_kw") is None and rg["cap"] is not None})
        row = {"model": m,
               "spurious_cap_values_kw": " ".join(str(v) for v in spur_vals),
               "ev_deadline_exact": f"{cnt['ev_ok']}/{cnt['ev_n']}",
               "power_cap_exact": f"{cnt['cap_ok']}/{cnt['cap_n']}",
               "spurious_cap": f"{cnt['cap_spurious']}/{cnt['nocap_n']}",
               "quiet_hours_exact": f"{cnt['quiet_ok']}/{cnt['quiet_n']}",
               "all_three_scheduled": f"{cnt['cover_ok']}/{cnt['cover_n']}"}
        ext_rows.append(row)
        say(f"  {m:10s} EV deadline {row['ev_deadline_exact']:>6s}  cap {row['power_cap_exact']:>5s}  "
            f"spurious cap {row['spurious_cap']:>6s}"
            + (f" ({row['spurious_cap_values_kw']} kW)" if row['spurious_cap_values_kw'] else "")
            + f"  quiet hours {row['quiet_hours_exact']:>4s}  "
            f"all three scheduled {row['all_three_scheduled']:>6s}")
    say("  (the household context line reads 'Household: 4 kWp rooftop solar', i.e. PV size, not a grid limit)")
    say("")

    say("Guard actions on the direct runs (pass / repair / block) and effect:")
    for m in MODELS5:
        ks = [k for k in out if k[0] == m]
        a = collections.Counter(out[k]["guard_action"] for k in ks)
        fixed = sum(1 for k in ks if not out[k]["direct"] and out[k]["guard"])
        broke = sum(1 for k in ks if out[k]["direct"] and not out[k]["guard"])
        say(f"  {m:10s} pass {a['pass']:2d}  repair {a['repair']:2d}  block {a['block']:2d}  "
            f"-> fixed {fixed:2d}, broke {broke:2d}")
    say("")

    # ---------------------------------------------------------- efficiency
    say("Efficiency per run, Exp 2 (direct-guided vs hybrid):")
    eff_rows = {}
    for m in MODELS5:
        parts = []
        for a, rows_ in (("direct", direct), ("hybrid", hybrid)):
            rs = [r for r in rows_ if r["model"] == m]
            tok = mean([r["tokens"]["input"] + r["tokens"]["output"] for r in rs])
            lat = mean([r["latency_s"] for r in rs])
            usd = sum(r["est_cost_usd"] or 0 for r in rs)
            n_ok = sum(out[k][a] for k in out if k[0] == m)
            eff_rows[(m, a)] = (tok, lat, usd / len(rs), usd / max(1, n_ok))
            parts.append(f"{a} {tok:7.0f} tok {lat:5.1f} s ${usd / max(1, n_ok):.4f}/correct")
        say(f"  {m:10s} " + "   ".join(parts))
    for r_ in arm_rows:
        for a in ("direct", "hybrid"):
            tok, lat, per_run, per_ok = eff_rows[(r_["model"], a)]
            r_[f"{a}_tokens"] = round(tok)
            r_[f"{a}_latency_s"] = round(lat, 1)
            r_[f"{a}_usd_per_correct"] = round(per_ok, 4)
    say("")

    # ================================================================ Exp 3
    sel = archive.load_selection()
    regime_of = {d: rg for rg, ds in sel["exp3_regimes"].items() for d in ds}
    base = json.loads((config.DATA / "results" / "baselines.json").read_text(encoding="utf-8"))["per_day"]
    h3 = load("exp3-hybrid")
    d3 = load("exp3")
    cost = collections.defaultdict(list)
    nc = collections.Counter()
    match_wa = collections.Counter()
    for r in h3:
        c_ = (r["score"] or {}).get("net_cost_realized")
        if c_ is None:
            nc[r["model"]] += 1
            continue
        cost[(r["model"], r["day"], "hybrid")].append(c_)
        if (r["score"]["committed_starts"] ==
                base[r["day"]]["wfh_ev18"]["oracle"]["starts"]):
            match_wa[r["model"]] += 1
    for r in d3:
        c_ = (r["score"] or {}).get("net_cost_realized")
        if c_ is not None:
            cost[(r["model"], r["day"], r["scenario"])].append(c_)
    arms3 = ["price_only", "weather_aware", "hybrid"]
    milp = {d: milp_arms(d) for d in regime_of}
    say(f"Exp 3: {len(h3)} hybrid runs; non-commits "
        + ", ".join(f"{m} {nc[m]}/45" for m in MODELS3)
        + "; schedule identical to the weather-aware MILP in "
        + ", ".join(f"{m} {match_wa[m]}/{45 - nc[m]}" for m in MODELS3))
    say("Realized net cost, GBP/day (agent arms matched over model-days present in all three):")
    say(f"  {'regime':9s} {'price-only agent':>17s} {'weather agent':>14s} {'hybrid':>8s} "
        f"{'PO MILP':>8s} {'WA MILP':>8s} {'hybrid captures':>16s}")
    e3_rows, units = [], []
    for reg in ("overcast", "mixed", "sunny"):
        pairs = [(m, d) for m in MODELS3 for d in sel["exp3_regimes"][reg]
                 if all(cost[(m, d, a)] for a in arms3)]
        v = {a: mean([mean(cost[(m, d, a)]) for m, d in pairs]) for a in arms3}
        po_m = mean([milp[d]["po_milp"]["cost"] for d in sel["exp3_regimes"][reg]])
        wa_m = mean([milp[d]["wa_milp"]["cost"] for d in sel["exp3_regimes"][reg]])
        cap_share = 100 * (po_m - v["hybrid"]) / (po_m - wa_m)
        say(f"  {reg:9s} {v['price_only']:17.3f} {v['weather_aware']:14.3f} {v['hybrid']:8.3f} "
            f"{po_m:8.3f} {wa_m:8.3f} {cap_share:15.0f}%")
        e3_rows.append({"regime": reg, "price_only_agent": round(v["price_only"], 3),
                        "weather_aware_agent": round(v["weather_aware"], 3),
                        "hybrid": round(v["hybrid"], 3), "price_only_milp": round(po_m, 3),
                        "weather_aware_milp": round(wa_m, 3),
                        "hybrid_weather_value_captured_pct": round(cap_share, 1),
                        "model_days": len(pairs)})
    for d in sorted(regime_of):
        ms = [m for m in MODELS3 if all(cost[(m, d, a)] for a in arms3)]
        if ms:
            units.append((mean([mean(cost[(m, d, "price_only")]) - mean(cost[(m, d, "hybrid")]) for m in ms]),
                          mean([mean(cost[(m, d, "price_only")]) - mean(cost[(m, d, "weather_aware")]) for m in ms])))
    h_est, a_est = mean([u[0] for u in units]), mean([u[1] for u in units])
    h_ci = boot(units, lambda u: mean([x[0] for x in u]))
    a_ci = boot(units, lambda u: mean([x[1] for x in u]))
    say("Value of the forecast relative to the price-only agent, pooled over 15 days "
        "(GBP/day; bootstrap over days):")
    say(f"  hybrid          {h_est:+.3f}  95% CI [{h_ci[0]:+.3f}, {h_ci[1]:+.3f}]")
    say(f"  weather agents  {a_est:+.3f}  95% CI [{a_ci[0]:+.3f}, {a_ci[1]:+.3f}]")
    tok3 = {a: mean([r["tokens"]["input"] + r["tokens"]["output"] for r in rs])
            for a, rs in (("weather_aware", [r for r in d3 if r["scenario"] == "weather_aware"]),
                          ("hybrid", h3))}
    usd3 = {a: mean([r["est_cost_usd"] or 0 for r in rs])
            for a, rs in (("weather_aware", [r for r in d3 if r["scenario"] == "weather_aware"]),
                          ("hybrid", h3))}
    say(f"  tokens per run: weather agents {tok3['weather_aware']:.0f}, hybrid {tok3['hybrid']:.0f}; "
        f"cost per run: ${usd3['weather_aware']:.4f} vs ${usd3['hybrid']:.4f}")

    # ---------------------------------------------------------- write
    def wcsv(name, rows_):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader(); w.writerows(rows_)
    wcsv("b3_exp2_arms.csv", arm_rows)
    wcsv("b3_exp2_by_family.csv", fam_rows)
    if tax_rows:
        wcsv("b3_hybrid_failures.csv", tax_rows)
    wcsv("b3_extraction.csv", ext_rows)
    wcsv("b3_exp3.csv", e3_rows)
    fig = []
    for m in MODELS5:
        for f in FAMILIES:
            ks = [k for k in out if k[0] == m and family(k[1]) == f]
            for a, label in (("direct", "direct"), ("hybrid", "hybrid")):
                rate = f"{mean([out[k][a] for k in ks]):.4f}"
                fig.append({"model": m, "prompt": label, "family": FAMKEY[f],
                            "success_rate": "" if f == "S4" else rate,
                            "infeasibility_reported_rate": rate if f == "S4" else ""})
    wcsv("b3_fig_direct_vs_hybrid.csv", fig)
    tex = ["% Exp 2 arms: model | direct | guard | hybrid (correct rate [95% CI]) | "
           "hybrid-direct (pp [CI]) | tokens direct/hybrid | $/correct direct/hybrid"]
    for r_ in arm_rows:
        tex.append(f"{r_['model']} & {100*r_['direct_rate']:.0f} {r_['direct_ci']} & "
                   f"{100*r_['guard_rate']:.0f} {r_['guard_ci']} & "
                   f"{100*r_['hybrid_rate']:.0f} {r_['hybrid_ci']} & "
                   f"{r_['hybrid_minus_direct_pp']:+.0f} {r_['diff_ci']} & "
                   f"{r_['direct_tokens']}/{r_['hybrid_tokens']} & "
                   f"{r_['direct_usd_per_correct']:.4f}/{r_['hybrid_usd_per_correct']:.4f} " + r"\\")
    (OUT / "b3_exp2_arms.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    tex3 = ["% Exp 3: regime | price-only agent | weather-aware agent | hybrid | PO MILP | "
            "WA MILP | hybrid weather value captured (%)"]
    for r_ in e3_rows:
        tex3.append(f"{r_['regime'].capitalize()} & {r_['price_only_agent']:.3f} & "
                    f"{r_['weather_aware_agent']:.3f} & {r_['hybrid']:.3f} & "
                    f"{r_['price_only_milp']:.3f} & {r_['weather_aware_milp']:.3f} & "
                    f"{r_['hybrid_weather_value_captured_pct']:.0f} " + r"\\")
    (OUT / "b3_exp3.tex").write_text("\n".join(tex3) + "\n", encoding="utf-8")
    (OUT / "b3_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote revision/outputs/b3_*.csv, b3_exp2_arms.tex, b3_exp3.tex, b3_summary.txt")


if __name__ == "__main__":
    main()
