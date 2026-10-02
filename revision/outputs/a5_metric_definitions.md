# Metric definitions (from experiments/scorer.py and analyze.py)

All definitions apply per run. J* is the MILP optimum for the same day, appliances
and constraints; J is the net cost of the committed schedule under the same
accounting.

- **Success**: every requested appliance is committed through the scheduling tool and
  the committed schedule violates no stated constraint (horizon, EV deadline,
  power cap, quiet hours). For infeasible requests (S4), success means reporting
  infeasibility without committing the impossible appliance.
- **Exact optimality**: all appliances committed and |J - J*| < GBP 1e-06
  (no run in Exp 1 is optimal without also being successful).
- **Near-optimality**: success and gamma <= 1%. (The submitted version did not
  require success, which counted 12 constraint-violating text-interface runs as
  near-optimal; see Table 4.)
- **gamma (relative cost gap)**: (J - J*) / |J*|, undefined when |J*| < 1e-9. In
  Exp 1, J* ranges from GBP 1.51 to 9.55, so gamma is always
  defined. The absolute gap J - J* (GBP) is reported beside it because gamma
  inflates on low-cost days (the largest gamma, 122.9%, is GBP 1.86).
  Median and maximum gaps are over successful runs.
- **$/success**: total estimated API inference cost of all runs in a cell (successful
  or not, at list prices) divided by the number of successful runs.
- **Deadline**: an EV deadline of HH:MM is met when charging finishes at or before
  HH:MM, whether the time is stated directly or taken from a calendar entry.

Uncertainty. Repeats run at temperature 0 and are strongly correlated (90% of Exp 1
cells and 82% of Exp 2 cells have identical outcomes across the three repeats).
Intervals therefore use the day (Exp 1) or scenario x prompt cell (Exp 2) as the sampling
unit: Wilson score intervals for proportions, and a percentile bootstrap over days
(10000 resamples) for paired and continuous comparisons.

Sampling. Candidate days are the 62 complete Octopus Agile days (48 half-hourly prices,
region C) from 2026-04-20 to 2026-06-20. Exp 1 ranks them by the coefficient of variation of
the import price and splits them into terciles at CoV = 0.274 and 0.308; four days per tercile
are taken evenly spaced in time, with a negative-price day swapped in if none is selected.
Exp 3 classifies days by forecast daytime (06-20 h) mean cloud cover: below 30% sunny, above
70% overcast, otherwise mixed; five days per regime, evenly spaced in time. Exp 4a uses the
first Monday-Sunday week fully covered by the archive.
