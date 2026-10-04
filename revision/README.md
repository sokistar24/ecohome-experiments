# Revision work (resubmission)

Scripts for the resubmission. They only read the existing archives and run
logs (`data/archive/`, `data/runs/`); nothing here calls an LLM unless stated.
Run every script from the repo root.

Outputs go to `revision/outputs/` (tracked in git, unlike `data/` and `figs/`).

## Phase A: corrections from existing logs ($0, no LLM calls)

| Step | Command | Status |
|---|---|---|
| A1 Rescore Exp 2 without the unstated 30-min departure buffer | `python -m revision.a1_rescore_departure_buffer` then `python plot_fig3_taxonomy.py --csv revision/outputs/a1_exp2_main_rescored.csv --out revision/outputs/fig3_taxonomy_rescored` | done |
| A2 Exp 3 re-analysis: MILP reference arms, matched agent comparison, Fig 5 rebuild | `python -m revision.a2_exp3_milp_reference` then `python plot_fig4_sensitivity.py --noise-csv revision/outputs/a2_fig5_matched.csv --out revision/outputs/fig5_sensitivity_matched --legend-loc below` | done |
| A3 Exp 4a: weather value captured, baseline feasibility, schedule-match diagnostic | `python -m revision.a3_exp4a_weather_value` | done |
| A4 Exp 2 violation severity (minutes late, kW over cap) | `python -m revision.a4_exp2_violation_severity` | done |
| A5 Metric definitions, absolute gaps, intervals, paired FC-vs-text tests, sampling text | `python -m revision.a5_metrics_and_intervals` | done |
| A6 Weekly planning (5.5.2): rebuild plans from tool calls, score against the weekly MILP | `python -m revision.a6_weekly_planning_check` | done |

## Paper changes logged so far (applied in Phase C)

**A1**
- Table 5, S3 column: GPT 0.67/0.67 -> 0.67/0.78; Gemini and Claude 0.67/0.67 -> 1.00/1.00.
- Aggregate compliance: Claude 92% -> 100%, Gemini 86% -> 94%, GPT 65% -> 67%;
  Qwen 72% and Llama 50% unchanged.
- Figure 3: replace with `revision/outputs/fig3_taxonomy_rescored.pdf`.
- Section 5.3: remove the S3b "reasoning-to-action gap" paragraph. All 13 affected
  runs finished exactly at the stated 06:30 departure.
- Methods / Appendix B: state that a deadline means "finished by the stated time"
  in every scenario.
- Appendix B: S3a does not bind (cheapest EV window ends 16:00, before the 19:00
  departure), so S3a cannot fail by violation; describe it as a calendar-reading
  case or drop the "conflict binds" claim for it.

**A2**
- Table 6: replace with `revision/outputs/a2_table6.tex`. Agent arms are now matched
  by model-day, and the table adds price-only, weather-aware (forecast) and
  perfect-foresight MILP arms plus non-commit counts.
- Section 5.4 results: weather information is worth GBP 0.09-0.22/day at the MILP level
  in every regime (0.215 on sunny days), but the direct agents capture none of it:
  matched weather-aware minus price-only differences are within +/-0.05/day in every
  regime.
- The regime-dependent pattern in the published Table 6 (cheaper on overcast, dearer
  on sunny) came from Gemini failing to commit in 12/45 weather-aware runs, which
  changed which days were averaged in that arm. Report the non-commits.
- Remove the negative-price explanation (only 1/5 sunny days had any negative slot).
  Do not use a self-consumption explanation either: agent SCR is below the
  weather-aware MILP's in every regime.
- Figure 4 (illustrative overcast day): check that it is not unrepresentative now
  that the matched overcast effect is about zero.
- Figure 5: the CSV behind the published figure has the +10% and +50% points swapped
  relative to the run logs (labels verified against run_id hashes), and its zero point
  is the 3-model pooled mean rather than the champion's own runs. The text values
  matched the logs; the figure did not.
- Figure 5 matched by day: +/-10% has no effect; +/-25-50% costs about GBP 0.06-0.08/day
  in both directions. Remove the "over-forecasting is more costly" and "conservative
  forecasts are a safer default" claims (Sec. 5.4, 6.2, Fig. 5 caption). Consider
  dropping Fig. 5 for one sentence.
- Abstract and conclusion: remove "weather-aware scheduling provides regime-dependent
  cost and PV self-consumption benefits".
- Decide before Phase B: add the hybrid arm on the Exp 3 days (about 135 runs).

**A3**
- Table 7: replace with `revision/outputs/a3_table7.tex`. It adds an EV-deadline
  column, a perfect-foresight row and a "weather value captured" column. That column
  is 0% for the price-only MILP and 100% for the weather-aware MILP; agents score
  GPT -2%, Gemini -29%, Claude -74%.
- Section 5.5.1: the 96.7-98.0% timer-to-oracle share is almost all price shifting.
  The price-only MILP alone reaches 98.1%, and the forecast is worth GBP 0.52 in the
  week. Say this explicitly; it also answers R1's point that this is the least
  interesting result.
- Immediate start and the greedy heuristic miss the 18:00 EV deadline on all 7 days.
  State that they are infeasible reference points, and remove the "54.8-55.5% less
  than greedy" and "65.3-65.9% less than immediate" comparisons, or label them as
  comparisons with infeasible policies.
- Gemini failed to commit on 3/21 passes (those days average its other passes).
- Schedule match: across Exp 4a and the Exp 3 weather-aware arm, no agent run ever
  reproduces the weather-aware MILP schedule on a day when it differs from the
  price-only one. GPT reproduces the price-only MILP exactly in 12/21 and 33/45 runs.
  This is direct evidence that the agents do not act on the PV forecast. It fits the
  window tool summing price only, but the cause is not tested.

**A4**
- Table 5: add the per-model columns from `revision/outputs/a4_table5_columns.tex`:
  invalid commits / committing runs, S4 fabrications, worst deadline overrun, worst
  cap excess.
- Invalid commits under the A1 rule: 52 of 343 committing runs. GPT 21, Qwen 16,
  Llama 14, Gemini 1, Claude 0.
- Severity is not marginal. Every deadline violation on a feasible scenario is 7.5-9.5 h
  late (S3b Qwen, S3c Llama, S5 GPT/Qwen/Gemini). Cap violations are 2.2 kW over the 9 kW
  cap (S2), apart from one Llama run at 0.4 kW. Llama also starts noisy appliances
  30 min early in S3c.
- Section 5.3 / Discussion: with S3b removed, the failure mode is not near-miss
  slot selection. The models either honour the constraint or ignore it entirely,
  e.g. the S2 GPT answers inspected do not mention the cap, and every 9-9.5 h overrun is the day's cheapest unconstrained EV window (10:00-16:00).

**A5**
- Methods: insert `revision/outputs/a5_metric_definitions.md` (definitions, uncertainty,
  sampling) before Table 4. This also answers R1's $/success point and R3's sampling point.
- Table 4: replace with `revision/outputs/a5_table4.tex`. It adds 95% intervals and GBP
  gaps, and corrects near-optimality to require success: Llama text 0.75 -> 0.67,
  Qwen text 0.81 -> 0.56.
- The 122.86% Qwen gap is GBP 1.86 on the lowest-cost day (J* = GBP 1.51). J* is never
  zero or negative in Exp 1.
- Function calling vs text (paired, bootstrap over days). The success difference is
  clearly above zero only for Claude (+50 pp [28, 72]); for the other four the interval
  reaches 0. The optimality difference is clearly positive for GPT (+69 pp [53, 83]),
  Gemini (+36 [14, 61]) and Claude (+53 [31, 78]), but not for Llama (+6 [-14, 25]) or
  Qwen (+25 [0, 53]). Rewrite "text-parsed actions reduce reliability for every model"
  to match: the direction holds for all five, the interval excludes zero for the three
  commercial models.
- Section 5.3: replace "mean cost gap of -152.9%" with the absolute figure (GPT's
  violating schedules are GBP 2.47 cheaper than the feasible optimum on average).
- Exp 2 aggregate compliance with intervals (unit = scenario x prompt): Claude 100%
  [87, 100], Gemini 94% [78, 98], Qwen 72% [53, 85], GPT 67% [47, 82], Llama 50% [32, 68].
- Exp 3 headline with uncertainty: the forecast is worth +GBP 0.133/day [0.063, 0.211] to the
  MILP and -0.026 [-0.058, 0.000] to the agents. The value left on the table is
  GBP 0.160/day [0.074, 0.258].
- day_selection.json records export_rate 0.15 while scoring uses 0.05; selection does not
  use it, so mention or ignore.

**A6**
- The published claims in 5.5.2 are not supported by the logs. Each one fails:
  - "Commits every requested cycle": 0/10 runs committed exactly what was requested.
    9/10 added unrequested cycles (EV charges on Saturday/Sunday, a third washing-machine
    run), and 1/10 omitted three cycles.
  - "Near-optimal": the committed plans cost 93-152% more than the weekly MILP. Even
    keeping only the requested cycles (the agent's best case), they cost 54-66% more.
  - "Shortfalls arise mainly from cycle-to-day allocation": day allocation explains only
    about 5% of the requested-cycle gap; about 95% is within-day timing. The agent starts
    the EV and dishwasher at 00:00 every day and does not use midday PV.
- Harness issues to state or fix: schedule_appliance keeps one commitment per appliance
  name (later commits overwrite), and analyze.py never scored Exp 4b. The request says
  EV "ready by 08:00" but the weekly MILP used 07:30; the MILP cost is identical either
  way for both weeks.
- Decision for Phase C: cut 5.5.2 and Appendix C, or keep two sentences reporting the
  negative result (it supports the case for a deterministic layer). Do not keep the
  current text.


## Phase B: hybrid arm (LLM extracts requirements, MILP schedules)

Code: `revision/hybrid.py` (tool, prompt, grids) and `revision/run_hybrid.py` (runner).
The original harness is untouched: runs go through `experiments.runner.execute()`, so
logs have the same schema, scoring and resume behaviour. `config.py` is unchanged,
so the config hash is unchanged.

Design (state in the paper's Methods):
- The hybrid agent gets two tools: `optimize_schedule` and `report_infeasibility`. It
  never sees prices and never picks a slot. It passes, per appliance, `not_before` /
  `finish_by` ('HH:MM') and an optional `power_cap_kw`; the tool solves the MILP and
  commits. If no feasible schedule exists, the tool records infeasibility itself and
  commits nothing.
- Each call adds or updates the appliances it lists (latest wins per appliance; latest
  non-null cap wins), then the MILP re-optimises all registered appliances jointly.
  The final schedule depends only on the requirements passed, not on call order.
  A lock serialises parallel tool calls.
- Prompt `v3.1-hybrid`: same role, appliance list, deadline rule and guided conflict-rule
  text as `v2-guided`, plus the time convention (times refer to the scheduled day, as in
  the direct prompt's slot convention). Only the workflow changes. Compare against
  direct-guided runs.
- Pilot history: the first pilot (`v3-hybrid`, logs kept as `*.pilot-v3.jsonl`) used
  "each call replaces the schedule". GPT-4o-mini issued one call per appliance in
  parallel, LangGraph ran them in threads, and the final schedule depended on thread
  timing (S1a, S6). The prompt also lacked the time convention: GPT and Qwen passed
  "not before 20:00" meaning the previous evening. Both were fixed in v3.1 before
  the full run.
- The optimizer objective matches each experiment's scoring: price-only for Exp 2,
  net cost with forecast PV and export for Exp 3.
- S6 (tool failure) injects the one-off failure into `optimize_schedule`.

| Step | Command | Status |
|---|---|---|
| B1 Offline self-test (no API calls) | `python -m revision.run_hybrid --selftest` | done |
| B1 Pilot (39 runs; rerun after the v3.1 fix) | `python -m revision.run_hybrid --exp exp2-hybrid --pilot --max-cost 2` and `--exp exp3-hybrid --pilot --max-cost 1` | done |
| B2 Full runs (195 + 135) | `python -m revision.run_hybrid --exp exp2-hybrid --max-cost 5`, `--exp exp3-hybrid --max-cost 5` | done: 330 runs, 0 errors, ~$2.03 |
| B3 Analysis: direct vs guard vs hybrid, failure causes, extraction accuracy, Exp 3 | `python -m revision.b3_hybrid_analysis`, then the `plot_fig3_taxonomy.py ... --left-prompt direct --right-prompt hybrid` command in its docstring | done |

**B3 (paper changes for Phase C)**
- New table `b3_exp2_arms.tex` (direct / guard / hybrid, 95% CIs, tokens, $/correct) and new
  figure `fig_direct_vs_hybrid.pdf`. These become the core of Section 5.3 or a new section.
- Methods: describe the guard. It checks each committed schedule against the constraints the
  same model extracted in structured form, repairs violations with the optimizer, blocks
  commits when those constraints are infeasible, and keeps the agent's own infeasibility
  reports.
- Findings (A1 rule; 39 paired runs per model):
  - Correct rate, direct -> hybrid: Llama 38% -> 90% (+51 pp [31, 69]); GPT 72 -> 82
    (+10 [-13, 33]); Gemini 97 -> 100; Claude 100 -> 100; Qwen 79 -> 64 (-15 [-41, 10]).
  - The optimizer removes action errors. Invalid commits on feasible scenarios: GPT 9 -> 0
    (its cap violations and 9 h overruns disappear). Llama's and Qwen's remaining 3 each are
    S5 runs where the model dropped the deadline before calling the optimizer.
  - It does not remove interpretation errors. Of 25 hybrid failures, 22 come from passing a
    wrong constraint and 3 are protocol errors (GPT: S6 failure not retried x2, no call on
    S4b). Qwen reads "4 kWp rooftop solar" as a 4 kW grid cap (7 runs) and invents EV
    earliest-start times. GPT adds a washing-machine deadline that makes S2 infeasible. Both
    open models drop the S5 departure under both architectures.
  - Guard: GPT 72 -> 87, Llama 38 -> 59, Qwen 79 -> 64. A validator is only as reliable as
    the constraints it checks: Qwen's spurious constraints block 10 valid direct schedules.
  - Efficiency: the hybrid uses 3.2-5.0k tokens per run vs 16-31k direct, and $/correct is
    4-16x lower. Latency is lower for four of five models.
  - Exp 3: the hybrid captures 96-100% of the forecast's value. Its schedule is identical to
    the weather-aware MILP in 134/135 runs, with 0 non-commits. Value vs the price-only agent:
    +GBP 0.139/day [0.069, 0.216]; the direct weather agents get -0.026 [-0.058, 0.000].
    Add a hybrid column to Table 6 (`b3_exp3.tex`).
- Narrative for Phase C: LLMs should interpret and optimisers should decide; interpretation
  is the remaining risk, and validation must check against the right constraints.


## Phase C: paper rewrite

| Step | Command | Status |
|---|---|---|
| C0 Paper assets (all tables and figures) | `python -m revision.c0_paper_assets`, then compile `revision/outputs/paper/assets_preview.tex` | done |
| C1 Methods | (LaTeX); numbers from `python -m revision.c1_coupling_check` | done |
| C2 Results, Discussion, Limitations | (LaTeX); drift numbers from `python -m revision.c2_drift_check` | done |
| C3 Introduction, contributions, Related Work | (LaTeX) | |
| C4 Abstract, title, conclusion, appendices, consistency pass | (LaTeX) | |

C0 reruns A1-A5 and B3, then writes LaTeX `tabular` blocks and PDF figures to
`revision/outputs/paper/`. Captions and labels stay in `article.tex`; copy the folder into
the LaTeX project as `results/revision/`.

| Asset | Paper element |
|---|---|
| `tab_exp1.tex` | Table: task completion and action interface (Exp 1) |
| `tab_exp2_arms.tex` | Table: direct / guard / hybrid under constraint conflict |
| `tab_hybrid_fail.tex` | Table: sources of hybrid failures |
| `tab_exp3.tex` | Table: weather value, agents vs hybrid vs MILP references |
| `tab_exp4a.tex` | Table: seven-day deployment |
| `tab_exp2_prompts.tex` | Appendix table: direct agent, baseline/guided prompt, severity |
| `fig_efficiency.pdf` | Fig. 2: correct behaviour vs cost per correct schedule, direct to hybrid |
| `fig_direct_vs_hybrid.pdf` | Fig. 3: failure rate by scenario family, direct vs hybrid |
| `fig_sunny_day.pdf` | Fig. 4: one sunny day, schedules of each arm (selection rule: the sunny day on which the forecast is worth most to the MILP; direct lanes show each model's modal schedule) |

Note for C2: the submitted Fig. 4 (`plot_fig_money.py`) drew hard-coded illustrative
schedules, not logged runs. The new Fig. 4 is drawn from the logs.

**C1 (Methods)**: new Sections 3-5 in `article.tex` (formulation and ground truth, three
placements of the LLM with the TikZ Fig. 1 `figs/fig_architecture.tex`, evaluation design).
The coupling numbers quoted in Section 3.2 come from `revision/c1_coupling_check.py`.
Optional provider-drift check (direct agent, guided prompt, repeat 0, rerun at hybrid time):
`python -m revision.run_hybrid --exp exp2-drift --max-cost 3` (65 runs, about $1.60).

**C2 (Results and Discussion)**: Section 6 (Results) and Section 7 (Discussion with
Limitations) rewritten around RQ1-RQ4. Appendix B corrected (S5 is a 07:00 departure; S3/S4
cases described; deadline rule; S3a does not bind). The weekly-planning appendix is removed;
a new appendix holds the baseline/guided table with severity. Provider drift: October reruns
of the direct agent (`data/runs/exp2-drift.jsonl`) agree with July in 86% of runs vs 88%
between July repeats (`revision/c2_drift_check.py`).
