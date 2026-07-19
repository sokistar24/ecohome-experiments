"""
Archive access layer: the ONLY way experiment code reads the frozen data.

Everything downstream (baselines, runner, scorer, scenario miner) goes
through these loaders, so format assumptions live in exactly one place.
All loaders validate shape (48 price slots / 24 weather hours) and raise
on anything malformed -- consistent with the no-silent-fallback rule.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from typing import Dict, List, Tuple

from experiments import config
from experiments.pv_model import hourly_to_pv_slots


def load_selection() -> Dict:
    path = config.ARCHIVE / "day_selection.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python -m experiments.fetch_archives` first")
    return json.loads(path.read_text())


@lru_cache(maxsize=256)
def load_prices(day: str) -> Tuple[float, ...]:
    """48 half-hourly import prices [GBP/kWh] for a local day."""
    path = config.PRICES_DIR / f"{day}.csv"
    with open(path) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["slot"]))
    if len(rows) != config.SLOTS_PER_DAY:
        raise ValueError(f"{path}: {len(rows)} slots, expected {config.SLOTS_PER_DAY}")
    if [int(r["slot"]) for r in rows] != list(range(config.SLOTS_PER_DAY)):
        raise ValueError(f"{path}: non-contiguous slot indices")
    return tuple(float(r["price_gbp_per_kwh"]) for r in rows)


@lru_cache(maxsize=256)
def load_weather(day: str, kind: str) -> Tuple[Tuple[float, ...], ...]:
    """kind in {'fc','actual'} -> (ghi_w_m2[24], temperature_c[24], cloud_pct[24])."""
    d = {"fc": config.WEATHER_FC_DIR, "actual": config.WEATHER_ACT_DIR}[kind]
    path = d / f"{day}.csv"
    with open(path) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["hour"]))
    if len(rows) != 24:
        raise ValueError(f"{path}: {len(rows)} hours, expected 24")
    ghi = tuple(float(r["ghi_w_m2"] or 0.0) for r in rows)
    temp = tuple(float(r["temperature_c"]) for r in rows)
    cloud = tuple(float(r["cloudcover_pct"] or 0.0) for r in rows)
    return ghi, temp, cloud


def pv_slots(day: str, kind: str) -> List[float]:
    """48 half-hourly PV kWh for a day; kind='fc' (agent/oracle input) or
    'actual' (scoring basis)."""
    ghi, temp, _ = load_weather(day, kind)
    return hourly_to_pv_slots(list(ghi), list(temp))


def week_prices(days: List[str]) -> List[float]:
    out: List[float] = []
    for d in days:
        out.extend(load_prices(d))
    return out


def week_pv(days: List[str], kind: str) -> List[float]:
    out: List[float] = []
    for d in days:
        out.extend(pv_slots(d, kind))
    return out


def all_benchmark_days(sel: Dict) -> List[str]:
    """Union of every day any experiment touches (deduplicated, sorted)."""
    days = set()
    for ds in sel["exp1_terciles"].values():
        days.update(ds)
    for ds in sel["exp3_regimes"].values():
        days.update(ds)
    for week in sel["exp4_weeks"]:
        days.update(week)
    return sorted(days)
