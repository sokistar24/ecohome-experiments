"""
G9: the experiment runner.

    python -m experiments.runner --exp probe [--model gpt]
    python -m experiments.runner --exp smoke --model gpt
    (exp1/exp2/exp3/exp3-noise/exp4a/exp4b arrive in the next drop)

Guarantees (RUNBOOK "mistake-proofing"):
  * deterministic run_id -> completed runs are skipped; rerunning any
    command is always safe; --force redoes.
  * probe must pass (marker file) before any other experiment will start.
  * every JSONL row carries schema, config hash, prompt version, and the
    provider-reported model snapshot; --dry-run prints the grid + cost
    estimate with zero API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from experiments import archive, config, scorer
from experiments.eval_agent import invoke_agent
from experiments.eval_prompts import (PROMPT_VERSION_BASELINE,
                                      SYSTEM_BASELINE, context_for)
from experiments.eval_tools import FULL_TOOLKIT, RUN

load_dotenv()

PROBE_MARKER = config.RUNS / "probe_passed.marker"
_CONFIG_HASH = hashlib.sha1(
    Path(config.__file__).read_bytes()).hexdigest()[:10]

STANDARD_Q = ("Please schedule my washing machine, dishwasher and EV "
              "charging for tomorrow to minimise my electricity cost, and "
              "schedule all three. The washing machine and dishwasher can "
              "run at any time tomorrow; I only need the EV finished "
              "charging by 07:30.")
SINGLE_Q = ("Please schedule my washing machine for tomorrow at the "
            "cheapest time, and schedule it.")


def run_id(**kw) -> str:
    return hashlib.sha1(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:12]


def _done_ids(log: Path) -> set:
    if not log.exists():
        return set()
    out = set()
    for line in log.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        # A run counts as done only if it completed WITHOUT error. Errored
        # rows (e.g. a transient 429 / provider overload from DeepInfra) are
        # deliberately NOT marked done, so simply re-invoking the same command
        # retries exactly the failed run_ids and skips the good ones. The last
        # (successful) row for a run_id wins if an earlier attempt errored.
        rid = row.get("run_id")
        if rid is None:
            continue
        if row.get("error"):
            out.discard(rid)
        else:
            out.add(rid)
    return out


def _append(log: Path, row: dict):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(row) + "\n")


def _est_cost(model_key: str, tokens: dict) -> float:
    pin, pout = config.PRICE_PER_MTOK.get(model_key, (0, 0))
    return round(tokens["input"] * pin / 1e6 + tokens["output"] * pout / 1e6, 6)


def execute(spec: dict, log: Path) -> dict:
    """One agent run: reset context -> invoke -> score -> log."""
    RUN.reset(spec["eval_date"],
              inject_price_failure=spec.get("inject_price_failure", False),
              pv_noise=spec.get("pv_noise"))
    t0 = time.time()
    try:
        res = invoke_agent(spec["model"], interface=spec["interface"],
                           system_prompt=spec["system_prompt"],
                           context=spec["context"],
                           question=spec["question"],
                           toolkit=spec["toolkit"])
    except Exception as e:
        res = {"final": "", "tokens": {"input": 0, "output": 0},
               "snapshot": None, "iterations": 0,
               "error": f"{type(e).__name__}: {e}"}
    latency = round(time.time() - t0, 2)

    score = None
    if spec.get("score") and not res["error"]:
        score = scorer.score_run(
            day=spec["day"], requested=spec["requested"],
            constraints=spec.get("constraints", {}),
            committed=dict(RUN.committed),
            infeasibility_report=RUN.infeasibility_report,
            expected_infeasible=spec.get("expected_infeasible", False),
            use_pv_objective=spec.get("use_pv_objective", False))

    row = {"schema": config.RUN_SCHEMA, "run_id": spec["run_id"],
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "exp": spec["exp"], "scenario": spec.get("scenario"),
           "model": spec["model"], "interface": spec["interface"],
           "model_snapshot": res["snapshot"],
           "prompt_version": spec["prompt_version"],
           "config_hash": _CONFIG_HASH,
           "eval_date": spec["eval_date"], "day": spec.get("day"),
           "question": spec["question"],
           "tool_calls": list(RUN.tool_calls),
           "committed": dict(RUN.committed),
           "infeasibility_report": RUN.infeasibility_report,
           "final": (res["final"] or "")[:config.FINAL_ANSWER_TRUNCATE],
           "tokens": res["tokens"], "iterations": res["iterations"],
           "latency_s": latency,
           "est_cost_usd": _est_cost(spec["model"], res["tokens"]),
           "error": res["error"], "score": score}
    _append(log, row)
    return row


# ---------------------------------------------------------------- probe
def exp_probe(models):
    """G1 verification: does each model treat EVAL_DATE as 'today'?"""
    sel = archive.load_selection()
    eval_day = sel["exp1_terciles"]["low"][0]
    prev = (date.fromisoformat(eval_day) - timedelta(days=1)).isoformat()
    expect = eval_day                     # tomorrow relative to prev
    log = config.RUNS / "probe.jsonl"
    results = {}
    for m in models:
        spec = {"exp": "probe", "model": m, "interface": "fc",
                "eval_date": prev,
                "system_prompt": "You answer questions precisely.",
                "prompt_version": "probe-v1",
                "context": context_for(prev),
                "question": ("What is tomorrow's date? Reply with only the "
                             "date in YYYY-MM-DD format."),
                "toolkit": [], "score": False}
        spec["run_id"] = run_id(exp="probe", model=m, d=prev)
        row = execute(spec, log)
        ok = (expect in (row["final"] or "")) and not row["error"]
        results[m] = ok
        print(f"[probe] {m}: {'PASS' if ok else 'FAIL'}  "
              f"answer={row['final']!r} error={row['error']}")
    if all(results.values()):
        PROBE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        PROBE_MARKER.write_text(json.dumps(
            {"models": sorted(results), "expect": expect,
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        print(f"[probe] all models PASS -> {PROBE_MARKER}")
    else:
        print("[probe] FAILURES above -- experiments stay locked.")


# ---------------------------------------------------------------- smoke
def exp_smoke(model: str):
    if model != "mock" and not PROBE_MARKER.exists():
        raise SystemExit("probe has not passed; run --exp probe first")
    sel = archive.load_selection()
    day = sel["exp1_terciles"]["low"][0]
    eval_date = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    log = config.RUNS / "smoke.jsonl"
    cases = [("smoke_single", SINGLE_Q, ["washing_machine"], {}),
             ("smoke_multi", STANDARD_Q,
              ["washing_machine", "dishwasher", "ev_charger"],
              {"ev_latest_finish": 15})]
    done = _done_ids(log)
    for scen, q, requested, constraints in cases:
        rid = run_id(exp="smoke", scen=scen, model=model, day=day)
        if rid in done:
            print(f"[smoke:{scen}] already complete (run_id {rid}) -- skipped; "
                  "use --force to redo")
            continue
        spec = {"exp": "smoke", "scenario": scen, "model": model,
                "interface": "fc", "eval_date": eval_date, "day": day,
                "system_prompt": SYSTEM_BASELINE,
                "prompt_version": PROMPT_VERSION_BASELINE,
                "context": context_for(eval_date), "question": q,
                "toolkit": FULL_TOOLKIT, "requested": requested,
                "constraints": constraints, "score": True}
        spec["run_id"] = rid
        row = execute(spec, log)
        s = row["score"] or {}
        print(f"[smoke:{scen}] success={s.get('success')} "
              f"optimal={s.get('optimal')} gap={s.get('cost_gap')} "
              f"committed={s.get('committed_starts')} "
              f"milp={s.get('milp_starts')} "
              f"tokens={row['tokens']} ${row['est_cost_usd']} "
              f"error={row['error']}")
    print(f"[smoke] log -> {log}")


def _load_champion() -> str:
    path = config.DATA / "results" / "champion.json"
    if not path.exists():
        raise SystemExit("champion.json missing -- run "
                         "`python -m experiments.analyze --champion` after "
                         "exp1 and exp2 complete.")
    return json.loads(path.read_text())["champion"]


def _grid_for(exp: str, args):
    from experiments import grids
    if exp == "exp1":
        return grids.exp1_specs(pilot=args.pilot)
    if exp in ("exp2", "exp3", "exp4a"):
        return grids.GRIDS[exp]()
    if exp == "exp3-noise":
        return grids.exp3_noise_specs(_load_champion())
    if exp == "exp4b":
        return grids.exp4b_specs(_load_champion())
    raise SystemExit(f"unknown grid {exp}")


def run_grid(exp: str, args):
    if not PROBE_MARKER.exists():
        raise SystemExit("probe has not passed; run --exp probe first")
    log = config.RUNS / f"{exp}.jsonl"
    specs = _grid_for(exp, args)
    if args.model:                       # optional single-model slice
        specs = [s for s in specs if s["model"] == args.model]
    done = set() if args.force else _done_ids(log)
    todo = [s for s in specs if s["run_id"] not in done]

    if args.status:
        print(f"[{exp}] {len(specs)-len(todo)}/{len(specs)} complete, "
              f"{len(todo)} remaining")
        return
    if args.dry_run:
        by_model = {}
        for s in specs:
            by_model[s["model"]] = by_model.get(s["model"], 0) + 1
        print(f"[{exp}] {len(specs)} runs ({len(todo)} remaining): {by_model}")
        print(f"[{exp}] interfaces present: "
              f"{sorted({s['interface'] for s in specs})}")
        print("[dry-run] no API calls made.")
        return

    print(f"[{exp}] running {len(todo)}/{len(specs)} "
          f"({len(specs)-len(todo)} already done)")
    spent = 0.0
    for i, s in enumerate(todo, 1):
        row = execute(s, log)
        spent += row["est_cost_usd"] or 0.0
        sc = row["score"] or {}
        flag = "" if not row["error"] else f" ERROR:{row['error'][:40]}"
        print(f"[{exp} {i}/{len(todo)}] {s['model']}/{s['interface']}/"
              f"{s.get('scenario')}/{s['day']} r{s.get('rep')} "
              f"gap={sc.get('cost_gap')} ok={sc.get('success')} "
              f"${spent:.3f}{flag}")
        if args.max_cost and spent > args.max_cost:
            print(f"[{exp}] --max-cost {args.max_cost} exceeded at "
                  f"${spent:.3f}; stopping cleanly (rerun to resume).")
            break
    print(f"[{exp}] done this invocation; spent ~${spent:.3f}; log -> {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["probe", "smoke", "exp1", "exp2", "exp3",
                             "exp3-noise", "exp4a", "exp4b"])
    ap.add_argument("--model", default=None,
                    help="model key (gpt|gemini|claude|mock); slices a grid")
    ap.add_argument("--pilot", action="store_true",
                    help="exp1 only: 3 days, 1 rep, fc only (18 runs)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="abort cleanly once estimated spend (USD) exceeds this")
    args = ap.parse_args()

    if args.exp == "probe":
        models = [args.model] if args.model else list(config.MODELS)
        exp_probe(models)
    elif args.exp == "smoke":
        if args.force and (config.RUNS / "smoke.jsonl").exists():
            (config.RUNS / "smoke.jsonl").unlink()
        exp_smoke(args.model or "gpt")
    else:
        run_grid(args.exp, args)


if __name__ == "__main__":
    main()
