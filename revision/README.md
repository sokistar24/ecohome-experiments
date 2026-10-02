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
