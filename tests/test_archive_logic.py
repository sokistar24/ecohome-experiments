"""Offline tests for pv_model and fetch_archives' pure selection logic."""
from datetime import date, timedelta

from experiments import config
from experiments.pv_model import apply_noise, hourly_to_pv_slots
from experiments.fetch_archives import (price_day_stats, select_exp1_days,
                                        select_exp3_days, select_exp4_weeks)


def bell_day(peak_ghi=750.0):
    ghi = [0.0] * 24
    for h in range(5, 21):
        ghi[h] = peak_ghi * max(0.0, 1 - abs(h - 13) / 8)
    temp = [10 + 8 * max(0.0, 1 - abs(h - 15) / 9) for h in range(24)]
    return ghi, temp


def test_pv_model_shape_and_magnitude():
    ghi, temp = bell_day()
    slots = hourly_to_pv_slots(ghi, temp)
    assert len(slots) == 48
    assert all(g >= 0 for g in slots)
    assert slots[0] == 0.0 and slots[47] == 0.0          # night
    peak = max(slots)
    # 4 kWp * (750/1000) * PR 0.8 * temp derate * 0.5h  -> ~1.1 kWh
    assert 0.9 < peak < 1.3, peak
    assert slots.index(peak) in range(24, 30)            # around 13:00
    daily = sum(slots)
    # 4 kWp, 16-h synthetic midsummer bell: ~18-19 kWh (UK annual mean is
    # ~9 kWh/day, but peak-summer days legitimately reach 20+).
    assert 12 < daily < 24, daily


def test_pv_temperature_derating():
    ghi, _ = bell_day()
    cool = hourly_to_pv_slots(ghi, [5.0] * 24)
    hot = hourly_to_pv_slots(ghi, [32.0] * 24)
    assert sum(hot) < sum(cool)


def test_noise_is_deterministic_and_clamped():
    ghi, temp = bell_day()
    g = hourly_to_pv_slots(ghi, temp)
    up = apply_noise(g, 0.25, +1)
    dn = apply_noise(g, 0.25, -1)
    assert up == apply_noise(g, 0.25, +1)
    assert all(a >= b for a, b in zip(up, g)) and all(b >= c >= 0 for b, c in zip(g, dn))


def _synthetic_stats(n=45, neg_day_idx=40):
    stats, d0 = {}, date(2026, 4, 20)
    for i in range(n):
        d = (d0 + timedelta(days=i)).isoformat()
        prices = [0.10 + 0.002 * ((i * 7 + s) % 30) * (i % 3 + 1) / 3
                  for s in range(48)]
        if i == neg_day_idx:
            prices[4:8] = [-0.03] * 4
        stats[d] = price_day_stats(prices)
    return stats


def test_exp1_selection_terciles_and_negative_guarantee():
    stats = _synthetic_stats()
    picked = select_exp1_days(stats, config.EXP1_DAYS_PER_TERCILE)
    days = [d for ds in picked.values() for d in ds]
    assert len(days) == len(set(days)) == 3 * config.EXP1_DAYS_PER_TERCILE
    # tercile ordering holds: every low-CoV pick below every high-CoV pick
    assert max(stats[d]["cov"] for d in picked["low"]) <= \
           min(stats[d]["cov"] for d in picked["high"])
    assert any(stats[d]["has_negative"] for d in days), \
        "negative-price day must be represented"


def test_exp3_regime_split():
    days = {(date(2026, 4, 20) + timedelta(days=i)).isoformat() for i in range(30)}
    cloud = {}
    for i, d in enumerate(sorted(days)):
        cloud[d] = [15.0, 50.0, 85.0][i % 3]
    picked = select_exp3_days(cloud, days, config.EXP3_DAYS_PER_REGIME)
    assert all(len(v) == config.EXP3_DAYS_PER_REGIME for v in picked.values())
    assert all(cloud[d] < 30 for d in picked["sunny"])
    assert all(cloud[d] > 70 for d in picked["overcast"])


def test_exp4_weeks_are_consecutive_mon_sun():
    valid = {(date(2026, 4, 20) + timedelta(days=i)).isoformat()
             for i in range(45) if i != 9}          # hole on day 9
    weeks = select_exp4_weeks(valid, config.EXP4B_WEEKS)
    assert len(weeks) == config.EXP4B_WEEKS
    for w in weeks:
        assert len(w) == 7
        assert date.fromisoformat(w[0]).weekday() == 0
        assert all((date.fromisoformat(w[i + 1]) - date.fromisoformat(w[i])).days == 1
                   for i in range(6))
        assert set(w) <= valid                       # the hole forced a skip
