# EcoHome Experiments — Runbook

Every command runs from the repo root as `python -m experiments.<module>`.
`eval_tools.py`, `eval_agent.py`, `pv_model.py`, `optimizer.py`, `scorer.py`
are **libraries** — never run directly. The only entry points are the ones
below, in this order. Do not skip a gate.

---

## Phase A — One-time setup (zero tokens)

| # | Command | Produces | Done when |
|---|---------|----------|-----------|
| A1 | `python -m experiments.fetch_archives` | `data/archive/prices/`, `weather_fc/`, `weather_actual/`, `day_selection.json` | prints day selections for exp1/exp3/exp4 |
| A2 | `python -m pytest tests/ -q` | — | **all tests pass** |
| A3 | `python -m experiments.compute_baselines` | `data/results/baselines.json` | oracle + price-only MILP + 3 rule baselines priced for **every** selected day and week |

A3 runs before any LLM call on purpose: if the MILP solves cleanly on every
archived day, the archives are well-formed end-to-end. It also means the
reference rows of every results table exist before a single token is spent.

**Gate A→B:** A1–A3 all succeeded. If A1 changed `day_selection.json` after
A3 ran, rerun A3.

---

## Phase B — Gates (pennies)

| # | Command | Purpose | Pass criterion |
|---|---------|---------|----------------|
| B1 | `python -m experiments.runner --exp probe` | date-pinning check, all 3 models | each model answers "tomorrow" = EVAL_DATE+1 → writes `data/runs/probe_passed.marker` |
| B2 | `python -m experiments.runner --exp smoke --model gpt` | 1 day, single+multi, FC, scored vs MILP | JSONL row shows committed slots, γ computed, tokens, latency, $ |

The runner **refuses** to start any experiment unless `probe_passed.marker`
exists. This is the October-14th bug, formally closed before money is spent.

**Gate B→C:** B1 pass for all three models; B2 JSONL inspected by eye once.

---

## Phase C — Experiment 1  → paper Table IV (`tab:exp1main`), per-regime table, Pareto figure

| # | Command | Runs | Notes |
|---|---------|------|-------|
| C1 | `python -m experiments.runner --exp exp1 --pilot` | 18 | all models, 1 day/tercile, 1 rep — sanity + cost extrapolation |
| C2 | `python -m experiments.runner --exp exp1` | 432 | full grid: 3 models × 2 interfaces × 12 days × {single, multi} × 3 reps |
| C3 | `python -m experiments.analyze --exp exp1` | 0 | writes `data/results/exp1_main.csv` (+ `.tex`), stats tests, `pareto.png` |

**Gate C→D:** C2 shows 432/432 complete (`runner --exp exp1 --status`).

---

## Phase D — Experiment 2  → Table V (`tab:exp2main`), taxonomy figure; champion inputs

| # | Command | Runs | Notes |
|---|---------|------|-------|
| D1 | `python -m experiments.mine_scenarios` | 0 | builds the 13 S1–S6 instances from the price archive, **MILP-verifies each** (S1 binding, S4 infeasible); writes `data/archive/scenarios/` |
| D2 | `python -m experiments.runner --exp exp2` | 234 | 3 models × 13 instances × {baseline, guided} × 3 reps, FC only |
| D3 | `python -m experiments.analyze --exp exp2` | 0 | Table V numbers + trace excerpts for taxonomy coding |
| D4 | `python -m experiments.analyze --champion` | 0 | applies the pre-registered rule to C3+D3 outputs, writes `data/results/champion.json` — **never edit this file by hand** |

**Gate D→E:** `champion.json` exists. Exp 3-noise and Exp 4b read it; they
refuse to run without it (prevents accidentally running the sweep on the
wrong model).

---

## Phase E — Experiment 3  → Table VI (`tab:exp3main`), sensitivity figure, money figure

| # | Command | Runs | Notes |
|---|---------|------|-------|
| E1 | `python -m experiments.runner --exp exp3` | 270 | 3 models × 15 days × {price-only, weather-aware} × 3 reps. Price-only arm: weather/PV tools removed from toolkit + price-only prompt variant — enforced by the runner, not by hoping |
| E2 | `python -m experiments.runner --exp exp3-noise` | 135 | champion only × 15 days × 3 noise levels × 3 reps (paired ±, deterministic) |
| E3 | `python -m experiments.analyze --exp exp3` | 0 | Table VI, `sensitivity.png`, `money_figure.png` (illustrative day) |

---

## Phase F — Experiment 4  → Table VII (`tab:exp4main`), cumulative-divergence figure

| # | Command | Runs | Notes |
|---|---------|------|-------|
| F1 | `python -m experiments.runner --exp exp4a` | 63 | 3 models × 7 consecutive days × 3 passes; one invocation per simulated evening |
| F2 | `python -m experiments.runner --exp exp4b` | 10 | champion × 2 weeks × 5 reps (T=336 weekly planning) |
| F3 | `python -m experiments.analyze --exp exp4` | 0 | Table VII incl. bootstrap 4-week projections, `cumulative.png` |

---

## Mistake-proofing (built into the runner — why single runs are safe)

- **Idempotent + resumable.** Every run has a deterministic
  `run_id = hash(exp, scenario, model, interface, day, rep, prompt_version)`.
  Re-invoking any command **skips** completed run_ids and finishes the rest —
  a crash or rate-limit mid-grid costs nothing; just run the same command
  again. `--force` is the only way to redo work.
- **`--dry-run` on every exp** prints the full grid and estimated cost with
  zero API calls. Use it before C2, E1, E2, F1.
- **`--status` on every exp** shows completed/remaining counts.
- **Budget guard.** `--max-cost 10` (USD) aborts cleanly if cumulative
  estimated spend exceeds the cap.
- **Config freeze.** Every JSONL row records schema version, a hash of
  `config.py`, the prompt version, and the provider's returned model
  snapshot. `analyze` refuses to aggregate rows with mixed config hashes —
  you cannot accidentally blend runs from different configurations.
- **Ordering enforced in code.** No experiment without the probe marker; no
  noise sweep or 4b without `champion.json`; no runner at all without
  `day_selection.json`.

## What maps to what (single glance)

| Paper artifact | Produced by |
|---|---|
| Table IV `tab:exp1main` + per-regime + stats | C3 |
| Pareto figure `fig:pareto` | C3 |
| Table V `tab:exp2main` | D3 |
| Taxonomy figure `fig:taxonomy` | D3 (+ manual coding of excerpts) |
| Champion selection (funnel paragraph) | D4 |
| Table VI `tab:exp3main` | E3 |
| Sensitivity figure `fig:sensitivity` | E3 |
| Money figure (illustrative day) | E3 |
| Table VII `tab:exp4main` + projections | F3 |
| Cumulative figure `fig:cumulative` | F3 |
| Rule-baseline / oracle rows in ALL tables | A3 |

Total LLM runs: 18 (pilot) + 432 + 234 + 270 + 135 + 63 + 10 = **1,162**.
Everything else is free.
