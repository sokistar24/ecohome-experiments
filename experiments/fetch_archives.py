"""
G2: build the frozen evaluation archives (run ONCE locally, then never again).

    python -m experiments.fetch_archives            # full fetch + selection
    python -m experiments.fetch_archives --select-only   # redo selection from
                                                         # already-fetched files

Produces, under data/archive/:
    prices/YYYY-MM-DD.csv          slot,valid_from_utc,price_gbp_per_kwh   (48 rows)
    weather_fc/YYYY-MM-DD.csv      hour,temperature_c,cloudcover_pct,ghi_w_m2 (24 rows)
    weather_actual/YYYY-MM-DD.csv  same columns, ERA5 reanalysis
    day_selection.json             frozen day sets for Exp 1/3/4 + per-day stats

Design rules (experiment integrity):
  * No mocks, no silent fallbacks -- any fetch failure raises.
  * Prices are kept at native half-hourly resolution (48 slots/local day);
    days with != 48 slots (DST or gaps) are archived but marked incomplete
    and excluded from selection.
  * Everything an experiment consumes is a file in data/archive/; live APIs
    are never touched again after this script succeeds.

APIs (all keyless):
  Octopus:    /v1/products/  and  .../standard-unit-rates/
  Open-Meteo: historical-forecast-api (forecasts as issued),
              archive-api (ERA5 actuals)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from statistics import mean, pstdev
from zoneinfo import ZoneInfo

import requests

from experiments import config

TZ = ZoneInfo(config.TIMEZONE)
OCTOPUS = "https://api.octopus.energy/v1"
HOURLY_VARS = "temperature_2m,cloudcover,shortwave_radiation"


# ---------------------------------------------------------------- utilities
def _daterange(start: str, end: str):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def _get(url: str, **params) -> dict:
    r = requests.get(url, params=params or None, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- Octopus
def resolve_agile_product(probe_day: str) -> tuple[str, str]:
    """Pick the Agile import product that actually has 48 rates for a day
    inside our window; return (product_code, tariff_code). Frozen config
    value wins if set and valid."""
    candidates = []
    if config.AGILE_PRODUCT:
        candidates.append(config.AGILE_PRODUCT)
    page = _get(f"{OCTOPUS}/products/", page_size=250, brand="OCTOPUS_ENERGY")
    discovered = [p["code"] for p in page.get("results", [])
                  if "AGILE" in p["code"].upper()
                  and p.get("direction", "IMPORT") == "IMPORT"]
    # newest product codes sort last lexically for AGILE-YY-MM-DD; try newest first
    candidates += sorted(set(discovered) - set(candidates), reverse=True)
    # legacy code known to hold long history:
    if "AGILE-FLEX-22-11-25" not in candidates:
        candidates.append("AGILE-FLEX-22-11-25")

    diagnostics = {}
    for code in candidates:
        tariff = f"E-1R-{code}-{config.AGILE_REGION}"
        try:
            rows = fetch_rates_window(code, tariff, probe_day, probe_day)
            n = len(rows.get(probe_day, []))
            diagnostics[code] = n
            print(f"[probe] {code}: {n} slots for {probe_day}")
            if n == 48:
                return code, tariff
        except requests.HTTPError as exc:
            diagnostics[code] = f"HTTP {exc.response.status_code}"
            print(f"[probe] {code}: HTTP {exc.response.status_code}")
    raise RuntimeError(
        f"No Agile product returns 48 half-hourly rates for {probe_day} "
        f"in region {config.AGILE_REGION}. Slot counts: {diagnostics}")


def fetch_rates_window(product: str, tariff: str,
                       start: str, end: str) -> dict[str, list]:
    """Fetch [start, end] inclusive; return {local_date: [(slot, valid_from_utc,
    gbp_per_kwh), ...] sorted by slot}. Slot index is local (Europe/London)."""
    # BST: the local day starts at 23:00Z of the previous calendar day, so
    # the UTC request window must open one day early; the local-date filter
    # below trims precisely to [start, end].
    p_from = (date.fromisoformat(start) - timedelta(days=1)).isoformat() + "T00:00:00Z"
    p_to = (date.fromisoformat(end) + timedelta(days=2)).isoformat() + "T00:00:00Z"
    url = f"{OCTOPUS}/products/{product}/electricity-tariffs/{tariff}/standard-unit-rates/"
    results, page_url, params = [], url, {
        "period_from": p_from, "period_to": p_to, "page_size": 1500}
    while page_url:
        data = _get(page_url, **(params or {}))
        results += data.get("results", [])
        page_url, params = data.get("next"), None

    by_day: dict[str, list] = defaultdict(list)
    for r in results:
        vf_utc = datetime.fromisoformat(r["valid_from"].replace("Z", "+00:00"))
        vf_loc = vf_utc.astimezone(TZ)
        d = vf_loc.date().isoformat()
        if not (start <= d <= end):
            continue
        slot = vf_loc.hour * 2 + vf_loc.minute // 30
        by_day[d].append((slot, r["valid_from"],
                          round(r["value_inc_vat"] / 100.0, 5)))
    for d in by_day:
        by_day[d] = sorted(set(by_day[d]))
    return by_day


# ---------------------------------------------------------------- Open-Meteo
def fetch_weather(base_url: str, start: str, end: str) -> dict[str, list]:
    """Return {date: [(hour, temp, cloud, ghi) x24]} from an Open-Meteo API."""
    data = _get(base_url,
                latitude=config.LAT, longitude=config.LON,
                start_date=start, end_date=end,
                hourly=HOURLY_VARS, timezone=config.TIMEZONE)
    h = data["hourly"]
    out: dict[str, list] = defaultdict(list)
    for i, ts in enumerate(h["time"]):
        d, hour = ts[:10], int(ts[11:13])
        out[d].append((hour,
                       h["temperature_2m"][i],
                       h["cloudcover"][i],
                       h["shortwave_radiation"][i]))
    return out


# ---------------------------------------------------------------- writing
def write_prices(by_day: dict[str, list]) -> None:
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    for d, rows in sorted(by_day.items()):
        with open(config.PRICES_DIR / f"{d}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slot", "valid_from_utc", "price_gbp_per_kwh"])
            w.writerows(rows)


def write_weather(by_day: dict[str, list], out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for d, rows in sorted(by_day.items()):
        with open(out_dir / f"{d}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hour", "temperature_c", "cloudcover_pct", "ghi_w_m2"])
            w.writerows(sorted(rows))


# ---------------------------------------------------------------- selection
# Pure functions over per-day stats so they are unit-testable offline.
def price_day_stats(prices: list[float]) -> dict:
    mu, sd = mean(prices), pstdev(prices)
    return {"mean": round(mu, 5), "std": round(sd, 5),
            "cov": round(sd / abs(mu), 4) if mu != 0 else None,
            "has_negative": any(p < 0 for p in prices),
            "n_slots": len(prices)}


def _evenly_spaced(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


def select_exp1_days(stats: dict[str, dict], per_tercile: int) -> dict:
    """Volatility terciles by CoV; per_tercile days each, evenly spaced in
    time; guarantee >=1 negative-price day is included if any exist."""
    complete = sorted(d for d, s in stats.items()
                      if s["n_slots"] == config.SLOTS_PER_DAY and s["cov"] is not None)
    ranked = sorted(complete, key=lambda d: stats[d]["cov"])
    n = len(ranked)
    terciles = {"low": ranked[: n // 3],
                "mid": ranked[n // 3: 2 * n // 3],
                "high": ranked[2 * n // 3:]}
    picked = {name: sorted(_evenly_spaced(sorted(days), per_tercile))
              for name, days in terciles.items()}
    chosen = {d for ds in picked.values() for d in ds}
    negatives = [d for d in complete if stats[d]["has_negative"]]
    if negatives and not any(stats[d]["has_negative"] for d in chosen):
        neg = negatives[0]
        for name, days in terciles.items():          # swap into its own tercile
            if neg in days:
                picked[name][-1] = neg
                picked[name] = sorted(set(picked[name]))
    return picked


def select_exp3_days(fc_cloud_daytime: dict[str, float],
                     valid_days: set, per_regime: int) -> dict:
    """Regimes from forecast daytime (06-20h) mean cloud cover."""
    regimes = {"sunny": [], "mixed": [], "overcast": []}
    for d in sorted(valid_days):
        c = fc_cloud_daytime.get(d)
        if c is None:
            continue
        key = "sunny" if c < 30 else ("overcast" if c > 70 else "mixed")
        regimes[key].append(d)
    return {k: sorted(_evenly_spaced(v, per_regime)) for k, v in regimes.items()}


def select_exp4_weeks(valid_days: set, n_weeks: int) -> list:
    """Consecutive Mon-Sun weeks fully covered by valid days."""
    days = sorted(valid_days)
    weeks = []
    for d in days:
        d0 = date.fromisoformat(d)
        if d0.weekday() != 0:
            continue
        week = [(d0 + timedelta(days=i)).isoformat() for i in range(7)]
        if all(x in valid_days for x in week):
            weeks.append(week)
        if len(weeks) == n_weeks:
            break
    return weeks


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.ARCHIVE_START)
    ap.add_argument("--end", default=config.ARCHIVE_END)
    ap.add_argument("--select-only", action="store_true",
                    help="skip fetching; rebuild day_selection.json from disk")
    args = ap.parse_args(argv)

    if not args.select_only:
        probe = (date.fromisoformat(args.start)
                 + (date.fromisoformat(args.end) - date.fromisoformat(args.start)) / 2
                 ).isoformat()
        product, tariff = resolve_agile_product(probe)
        print(f"[prices] product={product} tariff={tariff}")
        rates = fetch_rates_window(product, tariff, args.start, args.end)
        write_prices(rates)
        print(f"[prices] wrote {len(rates)} day files")

        fc = fetch_weather(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            args.start, args.end)
        write_weather(fc, config.WEATHER_FC_DIR)
        print(f"[weather_fc] wrote {len(fc)} day files")

        act = fetch_weather(
            "https://archive-api.open-meteo.com/v1/archive",
            args.start, args.end)
        write_weather(act, config.WEATHER_ACT_DIR)
        print(f"[weather_actual] wrote {len(act)} day files")
    else:
        product = tariff = "(from previous fetch)"

    # ---- selection from disk (works for --select-only too) ---------------
    stats, fc_cloud = {}, {}
    for f in sorted(config.PRICES_DIR.glob("*.csv")):
        with open(f) as fh:
            prices = [float(r["price_gbp_per_kwh"]) for r in csv.DictReader(fh)]
        stats[f.stem] = price_day_stats(prices)
    for f in sorted(config.WEATHER_FC_DIR.glob("*.csv")):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        day = [float(r["cloudcover_pct"]) for r in rows
               if 6 <= int(r["hour"]) <= 20]
        fc_cloud[f.stem] = round(mean(day), 1) if day else None

    valid = {d for d, s in stats.items()
             if s["n_slots"] == config.SLOTS_PER_DAY
             and d in fc_cloud
             and (config.WEATHER_ACT_DIR / f"{d}.csv").exists()}
    print(f"[select] {len(valid)} fully-covered days")

    selection = {
        "generated_at": datetime.now(TZ).isoformat(),
        "product": product, "tariff": tariff,
        "region": config.AGILE_REGION,
        "export_rate_gbp": config.EXPORT_RATE_GBP,
        "window": [args.start, args.end],
        "exp1_terciles": select_exp1_days(
            {d: stats[d] for d in valid}, config.EXP1_DAYS_PER_TERCILE),
        "exp3_regimes": select_exp3_days(fc_cloud, valid,
                                         config.EXP3_DAYS_PER_REGIME),
        "exp4_weeks": select_exp4_weeks(valid, config.EXP4B_WEEKS),
        "negative_price_days": sorted(d for d in valid
                                      if stats[d]["has_negative"]),
        "day_stats": stats,
        "fc_daytime_cloud": fc_cloud,
    }
    out = config.ARCHIVE / "day_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2))
    print(f"[select] wrote {out}")
    for k in ("exp1_terciles", "exp3_regimes", "exp4_weeks"):
        print(f"  {k}: {json.dumps(selection[k])}")


if __name__ == "__main__":
    sys.exit(main())