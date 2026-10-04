"""
Phase B runner for the hybrid arm. Reuses experiments.runner.execute(), so
every row has the same schema, scoring and resume behaviour as the original
experiments. Logs: data/runs/exp2-hybrid.jsonl, data/runs/exp3-hybrid.jsonl.

Offline self-test (no API calls, no keys needed):
    python -m revision.run_hybrid --selftest

Pilot, then full grid (API calls; resumable, completed run_ids are skipped):
    python -m revision.run_hybrid --exp exp2-hybrid --pilot --dry-run
    python -m revision.run_hybrid --exp exp2-hybrid --pilot --max-cost 2
    python -m revision.run_hybrid --exp exp2-hybrid --max-cost 10
    python -m revision.run_hybrid --exp exp3-hybrid --max-cost 5
    python -m revision.run_hybrid --exp exp2-hybrid --status

Pilot runs are rep 0 of the full grid, so the full run skips them.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st

from experiments import config
from experiments import runner as R
from experiments.eval_tools import RUN
from revision.hybrid import GRIDS, HYBRID, _eval_date  # noqa: F401

SELFTEST_LOG = config.ROOT / "revision" / "outputs" / "b1_selftest.jsonl"


def _log_path(exp: str):
    return config.RUNS / f"{exp}.jsonl"


def _run(spec: dict, log):
    HYBRID.reset()
    HYBRID.use_pv = bool(spec.get("hybrid_use_pv"))
    return R.execute(spec, log)


def _direct_cost_reference(exp: str) -> dict:
    """Mean logged cost per run of the matching DIRECT arm, per model, as an
    upper-bound budget estimate (the hybrid prompt needs no price table)."""
    path, keep = ((config.RUNS / "exp2.jsonl", lambda r: r["prompt_version"] == "v2-guided")
                  if exp == "exp2-hybrid" else
                  (config.RUNS / "exp3.jsonl", lambda r: r["scenario"] == "weather_aware"))
    by = collections.defaultdict(list)
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if keep(r):
                by[r["model"]].append((r["est_cost_usd"] or 0, r["latency_s"] or 0))
    return {m: (st.mean(c for c, _ in v), st.mean(l for _, l in v)) for m, v in by.items()}


def run_grid(args):
    log = _log_path(args.exp)
    specs = GRIDS[args.exp](pilot=args.pilot)
    if args.model:
        specs = [s for s in specs if s["model"] == args.model]
    done = set() if args.force else R._done_ids(log)
    todo = [s for s in specs if s["run_id"] not in done]
    label = f"{args.exp}{' pilot' if args.pilot else ''}"
    if args.status:
        print(f"[{label}] {len(specs) - len(todo)}/{len(specs)} complete, {len(todo)} remaining")
        return
    if args.dry_run:
        ref = _direct_cost_reference(args.exp)
        by = collections.Counter(s["model"] for s in todo)
        cost = sum(n * ref.get(m, (0, 0))[0] for m, n in by.items())
        secs = sum(n * ref.get(m, (0, 0))[1] for m, n in by.items())
        print(f"[{label}] {len(specs)} runs, {len(todo)} remaining: {dict(by)}")
        print(f"[{label}] upper-bound estimate from the direct arm's logged runs: "
              f"~${cost:.2f}, ~{secs / 60:.0f} min sequential (hybrid should be cheaper)")
        print("[dry-run] no API calls made.")
        return
    print(f"[{label}] running {len(todo)}/{len(specs)} ({len(specs) - len(todo)} already done)")
    spent = 0.0
    for i, s in enumerate(todo, 1):
        row = _run(s, log)
        spent += row["est_cost_usd"] or 0.0
        sc = row["score"] or {}
        n_opt = sum(c["name"] == "optimize_schedule" for c in row["tool_calls"])
        flag = "" if not row["error"] else f" ERROR:{row['error'][:60]}"
        print(f"[{label} {i}/{len(todo)}] {s['model']}/{s['scenario']}/{s['day']} r{s['rep']} "
              f"optimize_calls={n_opt} ok={sc.get('success')} gap={sc.get('cost_gap')} "
              f"${spent:.3f}{flag}")
        if args.max_cost and spent > args.max_cost:
            print(f"[{label}] --max-cost {args.max_cost} exceeded at ${spent:.3f}; "
                  "stopping cleanly (rerun to resume).")
            break
    print(f"[{label}] done this invocation; spent ~${spent:.3f}; log -> {log}")


# ---------------------------------------------------------------- self-test
def _perfect_extraction_mock(spec):
    """Stands in for an LLM that extracts the scenario's TRUE constraints.
    Exercises the real tool, runner, logging and scorer end to end."""
    from experiments.eval_tools import _slot_time

    def _invoke(model_key, *, interface, system_prompt, context, question, toolkit):
        tools = {t.name: t for t in toolkit}
        cons = spec["constraints"]
        a = {"date": "tomorrow"}
        for name in spec["requested"]:
            a[name] = True
        if cons.get("ev_latest_finish") is not None:
            a["ev_charger_finish_by"] = _slot_time(cons["ev_latest_finish"])
        if cons.get("noisy_earliest_start"):
            a["washing_machine_not_before"] = _slot_time(cons["noisy_earliest_start"])
            a["dishwasher_not_before"] = _slot_time(cons["noisy_earliest_start"])
        if cons.get("power_cap_kw"):
            a["power_cap_kw"] = cons["power_cap_kw"]
        res = tools["optimize_schedule"].invoke(a)
        if "error" in res:                                  # S6: retry once
            tools["optimize_schedule"].invoke(a)
        return {"final": "mock", "tokens": {"input": 0, "output": 0},
                "snapshot": "mock", "iterations": 1, "error": None}
    return _invoke


def selftest():
    import copy
    from experiments.compute_baselines import EV_WFH_LATEST_FINISH  # noqa: F401
    base = json.loads((config.DATA / "results" / "baselines.json").read_text(encoding="utf-8"))["per_day"]
    if SELFTEST_LOG.exists():
        SELFTEST_LOG.unlink()
    original = R.invoke_agent
    rows = []
    try:
        for exp in ("exp2-hybrid", "exp3-hybrid"):
            seen = set()
            for s in GRIDS[exp]():
                key = (s["scenario"], s["day"])
                if s["rep"] != 0 or key in seen:
                    continue
                seen.add(key)
                spec = copy.deepcopy({k: v for k, v in s.items() if k != "toolkit"})
                spec["toolkit"] = s["toolkit"]
                spec["model"] = "mock"
                spec["run_id"] = f"selftest-{exp}-{s['scenario']}-{s['day']}"
                R.invoke_agent = _perfect_extraction_mock(spec)
                rows.append((exp, spec, _run(spec, SELFTEST_LOG)))
    finally:
        R.invoke_agent = original

    fails = []
    print("Self-test: perfect-extraction mock through the real tool, runner and scorer")
    for exp, spec, row in rows:
        sc = row["score"] or {}
        n_calls = sum(c["name"] == "optimize_schedule" for c in row["tool_calls"])
        if exp == "exp2-hybrid":
            if spec["expected_infeasible"]:
                ok = sc.get("infeasibility_reported") and not sc.get("fabricated_schedule")
            else:
                ok = sc.get("success") and sc.get("optimal")
            if spec.get("inject_price_failure"):
                ok = ok and n_calls == 2
            print(f"  {exp} {spec['scenario']:4s} correct={bool(ok)} optimize_calls={n_calls}")
        else:
            ref = base[spec["day"]]["wfh_ev18"]["oracle"]["net_cost_realized"]
            got = sc.get("net_cost_realized")
            ok = sc.get("success") and got is not None and abs(got - ref) < 1e-3
            print(f"  {exp} {spec['day']} realized {got} vs weather-aware MILP {ref} "
                  f"match={bool(ok)}")
        if not ok:
            fails.append((exp, spec["scenario"], spec["day"]))
    # ---- order independence under parallel per-appliance calls
    import itertools
    from concurrent.futures import ThreadPoolExecutor
    from experiments.eval_tools import RUN as _RUN
    from revision.hybrid import optimize_schedule, exp2_hybrid_specs
    by_id = {s["scenario"]: s for s in exp2_hybrid_specs() if s["rep"] == 0
             and s["model"] == "gpt"}
    for sid in ("S1a", "S2a"):
        spec = by_id[sid]
        cons = spec["constraints"]
        from experiments.eval_tools import _slot_time
        calls = [{"date": "tomorrow", "washing_machine": True},
                 {"date": "tomorrow", "dishwasher": True},
                 {"date": "tomorrow", "ev_charger": True,
                  "ev_charger_finish_by": _slot_time(cons["ev_latest_finish"])}]
        if cons.get("power_cap_kw"):
            for c in calls:
                c["power_cap_kw"] = cons["power_cap_kw"]
        joint = dict(calls[0]); joint.update(calls[1]); joint.update(calls[2])
        _RUN.reset(spec["eval_date"]); HYBRID.reset()
        optimize_schedule.invoke(joint)
        want = {a: v["slot"] for a, v in _RUN.committed.items()}
        outcomes = set()
        for perm in itertools.permutations(calls):          # sequential orders
            _RUN.reset(spec["eval_date"]); HYBRID.reset()
            for c in perm:
                optimize_schedule.invoke(c)
            outcomes.add(tuple(sorted((a, v["slot"]) for a, v in _RUN.committed.items())))
        for _ in range(20):                                  # threaded, as LangGraph does
            _RUN.reset(spec["eval_date"]); HYBRID.reset()
            with ThreadPoolExecutor(3) as ex:
                list(ex.map(optimize_schedule.invoke, calls))
            outcomes.add(tuple(sorted((a, v["slot"]) for a, v in _RUN.committed.items())))
        ok = outcomes == {tuple(sorted(want.items()))}
        print(f"  order independence {sid}: 3 single-appliance calls in 6 orders + "
              f"20 threaded runs -> {len(outcomes)} distinct final schedule(s), "
              f"equal to one joint call: {ok}")
        if not ok:
            fails.append(("order", sid, ""))

    print(f"Self-test log: {SELFTEST_LOG.relative_to(config.ROOT)}")
    if fails:
        raise SystemExit(f"SELF-TEST FAILED for {fails}")
    print(f"SELF-TEST PASSED: {len(rows)} runs. With correct extraction the hybrid arm "
          "is exactly optimal on every feasible Exp 2 scenario, reports every S4 "
          "infeasibility, retries the S6 failure, and reproduces the weather-aware "
          "MILP on all 15 Exp 3 days; split or parallel calls give the same final "
          "schedule as one joint call.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=sorted(GRIDS))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="exp2-hybrid: S1a,S2a,S3b,S4a,S5,S6 rep 0 (30 runs); "
                         "exp3-hybrid: one day per regime rep 0 (9 runs)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-cost", type=float, default=None)
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.exp:
        run_grid(args)
    else:
        ap.error("give --selftest or --exp")


if __name__ == "__main__":
    main()
