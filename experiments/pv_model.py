"""
PV generation model (paper Eq. (3), parameters frozen in config).

Maps an hourly weather forecast (global horizontal irradiance in W/m^2 and
ambient temperature in degC) to expected half-hourly PV generation in kWh:

    T_cell = T_amb + I * (NOCT - 20) / 800
    P_kW   = P_stc * (I / 1000) * [1 - alpha * (T_cell - 25)] * PR
    G_slot = P_kW * SLOT_HOURS          (each hour -> two identical slots)

Used by the `predict_pv_generation` eval tool (what the agent sees, from
the *forecast* archive) and by the scorer (realized generation, from the
*actuals* archive) -- same physics, different weather input, which is
exactly the forecast-vs-realized distinction Experiment 3 measures.
"""
from __future__ import annotations

from typing import List, Sequence

from experiments import config


def hourly_to_pv_slots(
    irradiance_w_m2: Sequence[float],
    temperature_c: Sequence[float],
    *,
    kwp: float = config.PV_KWP,
    pr: float = config.PV_PR,
    alpha: float = config.PV_ALPHA,
    noct: float = config.PV_NOCT,
    slot_hours: float = config.SLOT_HOURS,
) -> List[float]:
    """24 hourly (I, T) pairs -> 48 half-hourly generation values [kWh]."""
    assert len(irradiance_w_m2) == 24 and len(temperature_c) == 24, \
        "expected 24 hourly values"
    slots: List[float] = []
    for I, T_amb in zip(irradiance_w_m2, temperature_c):
        I = max(0.0, float(I or 0.0))
        T_amb = float(T_amb if T_amb is not None else 15.0)
        t_cell = T_amb + I * (noct - 20.0) / 800.0
        p_kw = kwp * (I / 1000.0) * (1.0 - alpha * (t_cell - 25.0)) * pr
        g = max(0.0, p_kw * slot_hours)
        slots.extend([round(g, 5), round(g, 5)])   # two half-hours per hour
    return slots


def apply_noise(pv_slots: Sequence[float], level: float, sign: int) -> List[float]:
    """Deterministic multiplicative perturbation for the Exp 3 sweep.

    level in {0.10, 0.25, 0.50}; sign in {+1, -1}. Applied uniformly so a
    given (level, sign) run is exactly reproducible; the paired +/- design
    brackets the effect of systematic over/under-forecasting.
    """
    f = 1.0 + sign * level
    return [max(0.0, g * f) for g in pv_slots]
