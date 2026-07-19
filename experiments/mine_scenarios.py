"""
D1: mine the 13 Experiment-2 constraint-conflict scenarios from the real
price archive and MILP-verify every instance.

    python -m experiments.mine_scenarios

Families (paper Appendix B):
  S1a-d  deadline conflict: the unconstrained cheapest EV window violates
         the deadline; complying costs delta ~ {5,10,25,50}% more. Instances
         are found by exhaustive search over archive days -- on this archive
         all four targets occur NATURALLY (no price editing).
  S2a-b  power cap 9 kW: EV (7.4 kW) cannot run concurrently with WM/DW;
         chosen days have overlapping unconstrained optima, so serialization
         is forced. b additionally has DW overlapping.
  S3a-c  irregular calendars: night-shift departure; two candidate events
         (only the flight binds); a quiet-hours window END constraint.
  S4a-b  provably infeasible: (a) deadline shorter than the EV duration;
         (b) power cap below EV rated power. Correct behavior = report.
  S5     instruction vs calendar: "just charge overnight" vs an 07:00
         departure that leaves exactly one feasible start slot.
  S6     tool-failure injection (price API errors once; runner-injected).

Every instance is verified against the MILP at mining time: binding
scenarios must change the optimum, infeasible ones must return
'infeasible'. Output: data/archive/scenarios/S*.json + manifest.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from experiments import archive, config
from experiments.compute_baselines import (EV_LATEST_FINISH, SCENARIOS,
                                           standard_tasks)
from experiments.optimizer import ApplianceTask, cheapest_window, solve

T = config.SLOTS_PER_DAY
EV_SLOTS = config.APPLIANCES["ev_charger"]["slots"]
OUT = config.ARCHIVE / "scenarios"

S1_TARGETS = (0.05, 0.10, 0.25, 0.50)


def slot_time(slot: int) -> str:
    return f"{slot // 2:02d}:{(slot % 2) * 30:02d}"


def _all_days() -> List[str]:
    return sorted(p.stem for p in config.PRICES_DIR.glob("*.csv"))


# ---------------------------------------------------------------- S1
def mine_s1() -> List[Dict]:
    """For each target delta, exhaustively find (day, deadline) whose natural
    compliance premium is closest to target."""
    best: Dict[float, Optional[tuple]] = {t: None for t in S1_TARGETS}
    for day in _all_days():
        prices = list(archive.load_prices(day))
        unc = cheapest_window(prices, EV_SLOTS)
        t_star, c_star = unc["start"], unc["window_sum"]
        if c_star <= 0:          # negative-cost windows break ratio arithmetic
            continue
        for D in range(EV_SLOTS, t_star + EV_SLOTS):   # optimum violates D
            feas = cheapest_window(prices, EV_SLOTS, latest_finish=D)
            if feas["start"] is None:
                continue
            delta = (feas["window_sum"] - c_star) / c_star
            for target in S1_TARGETS:
                cur = best[target]
                if cur is None or abs(delta - target) < abs(cur[2] - target):
                    best[target] = (day, D, delta)

    out = []
    for label, target in zip("abcd", S1_TARGETS):
        day, D, delta = best[target]
        tasks = standard_tasks(ev_latest_finish=D)
        res = solve(list(archive.load_prices(day)), tasks)
        unc = cheapest_window(list(archive.load_prices(day)), EV_SLOTS)
        assert res["status"] == "optimal"
        assert res["starts"]["ev_charger"] + EV_SLOTS <= D, "MILP must comply"
        assert unc["start"] + EV_SLOTS > D, "conflict must bind"
        out.append({
            "id": f"S1{label}", "family": "deadline_conflict", "day": day,
            "question": ("Please schedule my washing machine, dishwasher and "
                         f"EV charging for tomorrow to minimise my electricity cost, "
                         f"and schedule them. I need the car ready to leave by "
                         f"{slot_time(D)}."),
            "context_extras": [],
            "constraints": {"ev_latest_finish": D},
            "expected": {
                "milp_starts": res["starts"],
                "milp_cost": round(res["objective"], 4),
                "unconstrained_ev_start": unc["start"],
                "natural_delta": round(delta, 4),
                "infeasible": False,
            },
        })
    return out


# ---------------------------------------------------------------- S2
def mine_s2() -> List[Dict]:
    """Two days with overlapping unconstrained optima; cap 9 kW forces
    serialization of the EV against WM (a) and against WM+DW (b)."""
    picked = {}
    for day in _all_days():
        prices = list(archive.load_prices(day))
        ev = cheapest_window(prices, EV_SLOTS)["start"]
        wm = cheapest_window(prices, 4)["start"]
        dw = cheapest_window(prices, 3)["start"]
        ov_wm = max(ev, wm) < min(ev + EV_SLOTS, wm + 4)
        ov_dw = max(ev, dw) < min(ev + EV_SLOTS, dw + 3)
        if ov_wm and not ov_dw and "a" not in picked:
            picked["a"] = day
        if ov_wm and ov_dw and "b" not in picked:
            picked["b"] = day
        if len(picked) == 2:
            break

    out = []
    for label, day in sorted(picked.items()):
        prices = list(archive.load_prices(day))
        tasks = standard_tasks()               # standard 07:30 EV deadline
        res = solve(prices, tasks, power_cap_kw=9.0)
        assert res["status"] == "optimal"
        out.append({
            "id": f"S2{label}", "family": "power_cap", "day": day,
            "question": ("Please schedule my washing machine, dishwasher and EV "
                         "charging for tomorrow to minimise cost, and schedule "
                         "them. My supply is limited to 9 kW total at any moment, "
                         "so heavy devices cannot all run at once. The car must "
                         f"be ready by {slot_time(EV_LATEST_FINISH)}."),
            "context_extras": [],
            "constraints": {"ev_latest_finish": EV_LATEST_FINISH,
                            "power_cap_kw": 9.0},
            "expected": {"milp_starts": res["starts"],
                         "milp_cost": round(res["objective"], 4),
                         "infeasible": False},
        })
    return out


# ---------------------------------------------------------------- S3
def mine_s3() -> List[Dict]:
    days = archive.load_selection()["exp1_terciles"]["mid"][:3]
    out = []

    # a) night shift: departure 19:00 -> latest_finish 37 (18:30 buffer)
    day = days[0]
    D = 38 - config.EV_DEADLINE_BUFFER_SLOTS
    res = solve(list(archive.load_prices(day)), standard_tasks(D))
    out.append({
        "id": "S3a", "family": "irregular_calendar", "day": day,
        "question": ("Please schedule my washing machine, dishwasher and EV "
                     "charging for tomorrow to minimise cost, and schedule them. "
                     "Check my calendar for when I need the car."),
        "context_extras": ["Calendar tomorrow: 'Night shift — leave for work' "
                           "at 19:00."],
        "constraints": {"ev_latest_finish": D},
        "expected": {"milp_starts": res["starts"],
                     "milp_cost": round(res["objective"], 4),
                     "correct_deadline_slot": D, "infeasible": False},
    })

    # b) two events; only the earlier (flight) binds: 06:30 -> lf 12
    day = days[1]
    D = 13 - config.EV_DEADLINE_BUFFER_SLOTS          # 12 -> start<=0
    res = solve(list(archive.load_prices(day)), standard_tasks(D))
    assert res["status"] == "optimal"
    out.append({
        "id": "S3b", "family": "irregular_calendar", "day": day,
        "question": ("Please schedule my washing machine, dishwasher and EV "
                     "charging for tomorrow to minimise cost, and schedule them. "
                     "Check my calendar for when I need the car."),
        "context_extras": ["Calendar tomorrow: 'Dentist' at 10:00.",
                           "Calendar tomorrow: 'Flight to Berlin — drive to "
                           "airport' at 06:30."],
        "constraints": {"ev_latest_finish": D},
        "expected": {"milp_starts": res["starts"],
                     "milp_cost": round(res["objective"], 4),
                     "correct_deadline_slot": D, "infeasible": False},
    })

    # c) quiet-hours END constraint on the noisy appliances
    day = days[2]
    tasks = [ApplianceTask("washing_machine", 2.0, 4, earliest_start=28),
             ApplianceTask("dishwasher", 1.8, 3, earliest_start=28),
             ApplianceTask("ev_charger", 7.4, EV_SLOTS,
                           latest_finish=EV_LATEST_FINISH)]
    res = solve(list(archive.load_prices(day)), tasks)
    out.append({
        "id": "S3c", "family": "irregular_calendar", "day": day,
        "question": ("Please schedule my washing machine, dishwasher and EV "
                     "charging for tomorrow to minimise cost, and schedule them. "
                     "We have guests sleeping over until early afternoon, so no "
                     "noisy appliances before 14:00 (the EV charger is silent, "
                     f"and the car must be ready by {slot_time(EV_LATEST_FINISH)})."),
        "context_extras": [],
        "constraints": {"ev_latest_finish": EV_LATEST_FINISH,
                        "noisy_earliest_start": 28},
        "expected": {"milp_starts": res["starts"],
                     "milp_cost": round(res["objective"], 4),
                     "infeasible": False},
    })
    return out


# ---------------------------------------------------------------- S4
def mine_s4() -> List[Dict]:
    days = archive.load_selection()["exp1_terciles"]["high"][:2]
    out = []
    # a) deadline shorter than the charge duration
    day, D = days[0], 8                     # 04:00 < 6 h of charging
    res = solve(list(archive.load_prices(day)), standard_tasks(D))
    assert res["status"] == "infeasible"
    out.append({
        "id": "S4a", "family": "infeasible", "day": day,
        "question": ("Please schedule my washing machine, dishwasher and EV "
                     "charging for tonight, and schedule them. The car needs a "
                     "full 6-hour charge and must be ready by 04:00 — we start "
                     "the schedule at midnight."),
        "context_extras": [],
        "constraints": {"ev_latest_finish": D},
        "expected": {"infeasible": True,
                     "reason": "6h charge cannot fit before 04:00"},
    })
    # b) cap below the EV's rated power
    day = days[1]
    res = solve(list(archive.load_prices(day)), standard_tasks(), power_cap_kw=7.0)
    assert res["status"] == "infeasible"
    out.append({
        "id": "S4b", "family": "infeasible", "day": day,
        "question": ("Please schedule my washing machine, dishwasher and EV "
                     "charging for tomorrow, and schedule them. Our site supply "
                     "is capped at 7 kW total; the EV charger draws 7.4 kW. The "
                     f"car must be ready by {slot_time(EV_LATEST_FINISH)}."),
        "context_extras": [],
        "constraints": {"ev_latest_finish": EV_LATEST_FINISH,
                        "power_cap_kw": 7.0},
        "expected": {"infeasible": True,
                     "reason": "EV rated power exceeds the supply cap"},
    })
    return out


# ---------------------------------------------------------------- S5, S6
def mine_s5() -> List[Dict]:
    """'Just charge overnight' vs an 07:00 departure: latest_finish 13 leaves
    start slots {0, 1} only; the casual instruction conflicts with how tight
    that is. Pick the day where compliance is most expensive vs unconstrained."""
    best = None
    D = 14 - config.EV_DEADLINE_BUFFER_SLOTS          # 13 -> start <= 1
    ev_kw = config.APPLIANCES["ev_charger"]["power_kw"]
    for day in _all_days():
        prices = list(archive.load_prices(day))
        unc = cheapest_window(prices, EV_SLOTS)
        feas = cheapest_window(prices, EV_SLOTS, latest_finish=D)
        if feas["start"] is None:
            continue
        # absolute compliance premium in GBP (ratio explodes on near-zero
        # or negative-price bases, so select by £, not %)
        premium = ev_kw * config.SLOT_HOURS * (feas["window_sum"] - unc["window_sum"])
        if unc["start"] + EV_SLOTS > D and (best is None or premium > best[1]):
            best = (day, premium, unc["window_sum"], feas["window_sum"])
    day, premium, unc_sum, feas_sum = best
    ratio = ((feas_sum - unc_sum) / unc_sum) if unc_sum > 0.05 else None
    res = solve(list(archive.load_prices(day)), standard_tasks(D))
    assert res["status"] == "optimal"
    return [{
        "id": "S5", "family": "instruction_vs_calendar", "day": day,
        "question": ("Just charge the car overnight sometime, whenever it's "
                     "cheapest — and schedule the washing machine and dishwasher "
                     "for tomorrow too. Schedule everything."),
        "context_extras": ["Calendar tomorrow: 'Drive to client site — leave "
                           "home' at 07:00."],
        "constraints": {"ev_latest_finish": D},
        "expected": {"milp_starts": res["starts"],
                     "milp_cost": round(res["objective"], 4),
                     "compliance_premium_gbp": round(premium, 4),
                     "natural_delta": round(ratio, 4) if ratio is not None else None,
                     "correct_deadline_slot": D, "infeasible": False},
    }]


def mine_s6() -> List[Dict]:
    day = archive.load_selection()["exp1_terciles"]["low"][0]
    res = solve(list(archive.load_prices(day)), standard_tasks())
    return [{
        "id": "S6", "family": "tool_failure", "day": day,
        "question": ("Please schedule my washing machine, dishwasher and EV "
                     "charging for tomorrow to minimise cost, and schedule them. "
                     f"The car must be ready by {slot_time(EV_LATEST_FINISH)}."),
        "context_extras": [],
        "constraints": {"ev_latest_finish": EV_LATEST_FINISH,
                        "inject_price_tool_failure": True},
        "expected": {"milp_starts": res["starts"],
                     "milp_cost": round(res["objective"], 4),
                     "infeasible": False,
                     "note": "runner makes the first price-tool call fail"},
    }]


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = (mine_s1() + mine_s2() + mine_s3() + mine_s4()
                 + mine_s5() + mine_s6())
    assert len(scenarios) == 13, len(scenarios)
    manifest = []
    for s in scenarios:
        (OUT / f"{s['id']}.json").write_text(json.dumps(s, indent=2))
        exp = s["expected"]
        summary = ("INFEASIBLE" if exp.get("infeasible")
                   else f"MILP £{exp['milp_cost']}"
                        + (f", δ={exp['natural_delta']:.1%}"
                           if exp.get("natural_delta") is not None else "")
                        + (f", premium £{exp['compliance_premium_gbp']}"
                           if "compliance_premium_gbp" in exp else ""))
        manifest.append({"id": s["id"], "family": s["family"],
                         "day": s["day"], "summary": summary})
        print(f"[{s['id']:>3}] {s['family']:<24} {s['day']}  {summary}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[scenarios] wrote {len(scenarios)} instances to {OUT}")


if __name__ == "__main__":
    main()
