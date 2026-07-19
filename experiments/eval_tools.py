"""
Experiment toolkit: date-pinned, archive-backed tools (G1 + G4).

These SUPERSEDE the live tools in tools.py for experiments only -- the
deployed demo is untouched. Design rules:
  * Relative dates ("today"/"tomorrow") resolve against RUN.eval_date,
    never the wall clock. Archives are the only data source; no mocks, no
    fallbacks -- unknown dates return a tool error message.
  * schedule_appliance is the ONLY thing the scorer reads: prose does not
    count. It validates the horizon only -- deliberately NOT scenario
    deadlines/caps, because whether the agent respects those is exactly
    what Experiment 2 measures.
  * report_infeasibility gives the agent a formal channel for S4-type
    situations (correct behavior = report, never commit).
  * S6 injection: the first get_electricity_prices call fails once.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from experiments import archive, config
from experiments.pv_model import apply_noise, hourly_to_pv_slots

T = config.SLOTS_PER_DAY


class RunContext:
    """Mutable per-run state; the runner calls reset() before every run."""

    def __init__(self):
        self.reset("1970-01-01")

    def reset(self, eval_date: str, *, inject_price_failure: bool = False,
              pv_noise: Optional[tuple] = None):
        self.eval_date = eval_date                  # "today" for the agent
        self.inject_price_failure = inject_price_failure
        self.price_failures_served = 0
        self.pv_noise = pv_noise                    # (level, sign) or None
        self.committed: Dict[str, Dict] = {}        # appliance -> {slot,date}
        self.infeasibility_report: Optional[str] = None
        self.tool_calls: List[Dict] = []


RUN = RunContext()


def _resolve_date(date: str) -> str:
    from datetime import date as _d, timedelta
    d = (date or "tomorrow").strip().lower()
    if d in ("tomorrow", ""):
        return (_d.fromisoformat(RUN.eval_date) + timedelta(days=1)).isoformat()
    if d == "today":
        return RUN.eval_date
    return _d.fromisoformat(d).isoformat()          # raises on garbage


def _slot_time(slot: int) -> str:
    return f"{slot // 2:02d}:{(slot % 2) * 30:02d}"


def _log(name: str, args: Dict[str, Any]):
    RUN.tool_calls.append({"name": name, "args": args})


def _parse_start(start_time: Optional[str], start_slot: Optional[int]):
    if start_slot is not None:
        return int(start_slot)
    if start_time:
        hh, mm = start_time.strip().split(":")
        return int(hh) * 2 + (1 if int(mm) >= 30 else 0)
    return None


@tool
def get_electricity_prices(date: str = "tomorrow") -> Dict[str, Any]:
    """Get half-hourly electricity import prices (GBP/kWh) for a date.

    The day has 48 half-hour slots: slot s starts at (s//2):(00 or 30)
    local time -- slot 0 = 00:00, slot 15 = 07:30, slot 36 = 18:00.
    Prices can be negative. date: 'today', 'tomorrow', or YYYY-MM-DD."""
    _log("get_electricity_prices", {"date": date})
    if RUN.inject_price_failure and RUN.price_failures_served == 0:
        RUN.price_failures_served += 1
        return {"error": "pricing service temporarily unavailable (HTTP 503). "
                         "Please retry the request."}
    try:
        d = _resolve_date(date)
        prices = archive.load_prices(d)
    except Exception as e:
        return {"error": f"no price data for '{date}': {e}"}
    return {"date": d, "currency": "GBP", "unit": "per_kWh",
            "resolution": "half_hourly",
            "slots": [{"slot": s, "time": _slot_time(s), "price": p}
                      for s, p in enumerate(prices)]}


@tool
def get_weather_forecast(date: str = "tomorrow") -> Dict[str, Any]:
    """Get the hourly weather forecast for a date: temperature (degC),
    cloud cover (%), and solar irradiance GHI (W/m2). date: 'today',
    'tomorrow', or YYYY-MM-DD."""
    _log("get_weather_forecast", {"date": date})
    try:
        d = _resolve_date(date)
        ghi, temp, cloud = archive.load_weather(d, "fc")
    except Exception as e:
        return {"error": f"no forecast for '{date}': {e}"}
    return {"date": d,
            "hourly": [{"hour": h, "temperature_c": temp[h],
                        "cloudcover_pct": cloud[h], "ghi_w_m2": ghi[h]}
                       for h in range(24)]}


@tool
def predict_pv_generation(date: str = "tomorrow") -> Dict[str, Any]:
    """Predict the home's rooftop solar PV generation (kWh per half-hour
    slot, 48 slots) for a date from the weather forecast. Appliances run
    during generation use free solar power instead of imported
    electricity. date: 'today', 'tomorrow', or YYYY-MM-DD."""
    _log("predict_pv_generation", {"date": date})
    try:
        d = _resolve_date(date)
        ghi, temp, _ = archive.load_weather(d, "fc")
        pv = hourly_to_pv_slots(list(ghi), list(temp))
        if RUN.pv_noise:
            pv = apply_noise(pv, *RUN.pv_noise)
    except Exception as e:
        return {"error": f"no PV prediction for '{date}': {e}"}
    return {"date": d, "unit": "kWh_per_half_hour",
            "total_kwh": round(sum(pv), 2),
            "slots": [{"slot": s, "time": _slot_time(s), "kwh": round(g, 3)}
                      for s, g in enumerate(pv)]}


@tool
def calculate_window_sums(date: str = "tomorrow", window_slots: int = 12,
                          earliest_slot: int = 0,
                          latest_finish_slot: Optional[int] = None
                          ) -> Dict[str, Any]:
    """For a run of `window_slots` consecutive half-hour slots, total the
    price of every possible start within [earliest_slot,
    latest_finish_slot) and return the 5 cheapest and 3 most expensive
    windows. Run lengths: washing machine 4 slots, dishwasher 3, EV
    charger 12."""
    _log("calculate_window_sums", {"date": date, "window_slots": window_slots,
                                   "earliest_slot": earliest_slot,
                                   "latest_finish_slot": latest_finish_slot})
    try:
        d = _resolve_date(date)
        prices = archive.load_prices(d)
    except Exception as e:
        return {"error": f"no price data for '{date}': {e}"}
    lf = latest_finish_slot if latest_finish_slot is not None else T
    wins = []
    for t in range(max(0, int(earliest_slot)), min(T, lf) - int(window_slots) + 1):
        wins.append({"start_slot": t, "start_time": _slot_time(t),
                     "price_sum": round(sum(prices[t:t + int(window_slots)]), 5)})
    if not wins:
        return {"error": f"no window of {window_slots} slots fits in "
                         f"[{earliest_slot}, {lf})"}
    wins.sort(key=lambda w: w["price_sum"])
    return {"date": d, "window_slots": int(window_slots),
            "cheapest": wins[:5], "most_expensive": wins[-3:][::-1]}


@tool
def schedule_appliance(appliance: str, start_time: Optional[str] = None,
                       start_slot: Optional[int] = None,
                       date: str = "tomorrow") -> Dict[str, Any]:
    """COMMIT an appliance schedule. This is the only way to actually
    schedule -- text recommendations are not executed. appliance is one
    of: washing_machine (4 slots / 2h), dishwasher (3 slots / 1.5h),
    ev_charger (12 slots / 6h). Give start_time 'HH:MM' (on the half
    hour) or start_slot 0-47. Re-committing overwrites."""
    _log("schedule_appliance", {"appliance": appliance,
                                "start_time": start_time,
                                "start_slot": start_slot, "date": date})
    name = (appliance or "").strip().lower()
    if name not in config.APPLIANCES:
        return {"error": f"unknown appliance '{appliance}'; choose from "
                         f"{sorted(config.APPLIANCES)}"}
    s = _parse_start(start_time, start_slot)
    if s is None:
        return {"error": "provide start_time 'HH:MM' or start_slot 0-47"}
    try:
        d = _resolve_date(date)
    except Exception as e:
        return {"error": f"bad date '{date}': {e}"}
    slots = config.APPLIANCES[name]["slots"]
    if not (0 <= s <= T - slots):
        return {"error": f"start slot {s} invalid: {name} needs {slots} "
                         f"slots and must finish within the day (latest "
                         f"start {T - slots} = {_slot_time(T - slots)})"}
    RUN.committed[name] = {"slot": s, "date": d}
    end = s + slots
    return {"committed": True, "appliance": name, "date": d,
            "start_slot": s, "start_time": _slot_time(s),
            "end_time": _slot_time(end) if end < T else "24:00"}


@tool
def report_infeasibility(explanation: str) -> Dict[str, Any]:
    """Formally report that the requested schedule is IMPOSSIBLE under the
    stated constraints (e.g. a run cannot finish before its deadline, or
    the power cap can never accommodate a device). Use this INSTEAD of
    committing a schedule that violates constraints; explain which
    constraint fails and what would need to change."""
    _log("report_infeasibility", {"explanation": str(explanation)[:500]})
    RUN.infeasibility_report = str(explanation)
    return {"recorded": True}


FULL_TOOLKIT = [get_electricity_prices, get_weather_forecast,
                predict_pv_generation, calculate_window_sums,
                schedule_appliance, report_infeasibility]

PRICE_ONLY_TOOLKIT = [get_electricity_prices, calculate_window_sums,
                      schedule_appliance, report_infeasibility]
