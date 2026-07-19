# START HERE — EcoHome Experiments Package

**Version: v1** (2026-07-04). This zip is complete and self-contained.
If you have older zips (ecohome_m1_part1/2/3), DELETE them — everything
in them is in here.

## What this zip contains

```
experiments/            NEW code (added next to your agent.py/tools.py)
  config.py             all frozen experiment settings (models, tariff, PV, days)
  optimizer.py          the MILP ground-truth solver          [library]
  pv_model.py           irradiance -> PV generation           [library]
  fetch_archives.py     downloads price + weather data        [YOU RUN THIS]
  RUNBOOK.md            the full campaign plan (future steps — ignore for now)
tests/                  automated checks                      [YOU RUN THIS]
```

Nothing in this zip touches your existing files (agent.py, tools.py,
api.py, app.py, the database, the vector store). Your live demo is safe.

## What to run TODAY — two commands, in order

Open a terminal in your project folder (the one containing agent.py):

### Step 1 — verify the code works on your machine
```
pip install pulp pytest
python -m pytest tests/ -q
```
PASS looks like: `10 passed`. Anything else: stop, send the error back.

### Step 2 — download the experiment data (one time, ~1–2 minutes, no API keys)
```
python -m experiments.fetch_archives
```
PASS looks like:
```
[prices] product=AGILE-... tariff=E-1R-...
[prices] wrote ~62 day files
[weather_fc] wrote ~62 day files
[weather_actual] wrote ~62 day files
[select] ... days
[select] wrote data/archive/day_selection.json
```
If it crashes: stop, send the traceback back.

## Then send back these 4 files
1. `data/archive/day_selection.json`
2. one CSV from `data/archive/prices/`
3. one CSV from `data/archive/weather_fc/`
4. one CSV from `data/archive/weather_actual/`

## What happens after that
The next package (v2) will add the runner. Until you receive v2 there is
nothing else to run — every other command mentioned in RUNBOOK.md does not
exist yet. One command at a time, each announced explicitly.
