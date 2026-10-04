"""
Phase B -- the hybrid architecture: the LLM extracts requirements, a
deterministic MILP chooses and commits the schedule.

Direct arm (Experiments 1-4): the LLM reads prices, picks start slots and
commits each appliance with schedule_appliance.
Hybrid arm (this module): the LLM never sees prices and never picks a slot.
It translates the request into structured inputs and calls
optimize_schedule ONCE; the tool solves the same MILP the scorer uses as
ground truth and commits the result. If no feasible schedule exists the
tool records infeasibility itself and commits nothing.

Everything else is held fixed: same models, scenarios, context lines,
conflict-rule text (the guided prompt's), RunContext, runner.execute() and
scorer. Only the action layer and the workflow wording change.

The optimizer objective matches each experiment's scoring objective:
  exp2-hybrid  price-only (Exp 2 is scored price-only)
  exp3-hybrid  net cost with forecast PV and export (as Exp 3 is scored)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from experiments import archive, config
from experiments.eval_prompts import _GUIDANCE, context_for
from experiments.eval_tools import RUN, _log, _resolve_date, _slot_time, report_infeasibility
from experiments.optimizer import ApplianceTask, solve

T = config.SLOTS_PER_DAY
PROMPT_VERSION_HYBRID = "v3.1-hybrid"   # v3 pilot: replace semantics raced under parallel calls


class _HybridConfig:
    """Per-run state for the hybrid tool, kept outside RunContext so the
    original harness is untouched. The runner sets use_pv and calls reset()
    before every run.

    Requirements registry: each optimize_schedule call adds or updates the
    appliances it lists (latest call wins per appliance; latest non-null
    power cap wins), then the MILP re-optimises EVERY registered appliance
    jointly and the commitment is replaced by that joint result. The final
    state therefore depends only on the set of requirements passed, not on
    the order of calls. Some models (e.g. GPT-4o-mini) issue one call per
    appliance in parallel, and the LangGraph tool node runs parallel calls in
    threads, so the lock serialises them."""
    use_pv: bool = False

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.date = None
        self.reg = {}            # appliance -> {"not_before": slot, "finish_by": slot}
        self.cap = None


HYBRID = _HybridConfig()


def _to_slot(hhmm: Optional[str], *, finish: bool) -> Optional[int]:
    """'HH:MM' -> slot boundary. finish_by rounds DOWN to the half hour
    (conservative), not_before rounds UP. None/'' -> None."""
    if hhmm is None or str(hhmm).strip() == "":
        return None
    h, m = str(hhmm).strip().split(":")
    minutes = int(h) * 60 + int(m)
    if not 0 <= minutes <= 24 * 60:
        raise ValueError(f"time out of range: {hhmm}")
    return minutes // 30 if finish else -(-minutes // 30)


@tool
def optimize_schedule(date: str = "tomorrow",
                      washing_machine: bool = False,
                      washing_machine_not_before: Optional[str] = None,
                      washing_machine_finish_by: Optional[str] = None,
                      dishwasher: bool = False,
                      dishwasher_not_before: Optional[str] = None,
                      dishwasher_finish_by: Optional[str] = None,
                      ev_charger: bool = False,
                      ev_charger_not_before: Optional[str] = None,
                      ev_charger_finish_by: Optional[str] = None,
                      power_cap_kw: Optional[float] = None) -> Dict[str, Any]:
    """Compute and COMMIT the minimum-cost schedule with a deterministic
    optimizer. This is the only way to schedule appliances.

    Set washing_machine / dishwasher / ev_charger to true for every appliance
    to schedule. For each, optionally give not_before (earliest start) and
    finish_by (must have finished by), as 'HH:MM' local time on the
    scheduled day. power_cap_kw limits the combined power of these
    appliances running at the same time. date: 'today', 'tomorrow' or
    YYYY-MM-DD.

    All times refer to the scheduled day (00:00 to 24:00); a run cannot
    start before 00:00 of that day.

    Each call adds or updates the appliances it lists; the optimizer then
    re-optimises every appliance scheduled so far, jointly, and commits the
    result. Listing all appliances in one call is simplest. The most recent
    power_cap_kw given applies to all of them. If no schedule satisfies the
    constraints, the optimizer records that the request is infeasible and
    commits nothing."""
    with HYBRID.lock:
        return _optimize_locked(date, washing_machine, washing_machine_not_before,
                                washing_machine_finish_by, dishwasher,
                                dishwasher_not_before, dishwasher_finish_by,
                                ev_charger, ev_charger_not_before,
                                ev_charger_finish_by, power_cap_kw)


def _optimize_locked(date, washing_machine, washing_machine_not_before,
                     washing_machine_finish_by, dishwasher, dishwasher_not_before,
                     dishwasher_finish_by, ev_charger, ev_charger_not_before,
                     ev_charger_finish_by, power_cap_kw) -> Dict[str, Any]:
    args = {"date": date,
            "washing_machine": washing_machine,
            "washing_machine_not_before": washing_machine_not_before,
            "washing_machine_finish_by": washing_machine_finish_by,
            "dishwasher": dishwasher,
            "dishwasher_not_before": dishwasher_not_before,
            "dishwasher_finish_by": dishwasher_finish_by,
            "ev_charger": ev_charger,
            "ev_charger_not_before": ev_charger_not_before,
            "ev_charger_finish_by": ev_charger_finish_by,
            "power_cap_kw": power_cap_kw}
    _log("optimize_schedule", args)

    # S6: the tool-failure scenario injects one failure into the first call
    if RUN.inject_price_failure and RUN.price_failures_served == 0:
        RUN.price_failures_served += 1
        return {"error": "optimization service temporarily unavailable "
                         "(HTTP 503). Please retry the request."}

    requested = [a for a in config.APPLIANCES if args[a]]
    if not requested:
        return {"error": "no appliance selected; set washing_machine, "
                         "dishwasher and/or ev_charger to true"}
    try:
        d = _resolve_date(date)
        prices = list(archive.load_prices(d))
    except Exception as e:
        return {"error": f"no price data for '{date}': {e}"}
    try:
        update = {a: {"not_before": _to_slot(args[f"{a}_not_before"], finish=False),
                      "finish_by": _to_slot(args[f"{a}_finish_by"], finish=True)}
                  for a in requested}
    except Exception as e:
        return {"error": f"could not read a time ('HH:MM' expected): {e}"}

    if HYBRID.date != d:                     # a different day starts a new plan
        HYBRID.date, HYBRID.reg, HYBRID.cap = d, {}, None
    HYBRID.reg.update(update)
    if power_cap_kw is not None:
        HYBRID.cap = power_cap_kw
    tasks = [ApplianceTask(a, config.APPLIANCES[a]["power_kw"],
                           config.APPLIANCES[a]["slots"],
                           earliest_start=c["not_before"] or 0,
                           latest_finish=c["finish_by"])
             for a, c in sorted(HYBRID.reg.items())]

    pv = archive.pv_slots(d, "fc") if HYBRID.use_pv else None
    res = solve(prices, tasks, pv=pv,
                export_rate=config.EXPORT_RATE_GBP if pv else 0.0,
                power_cap_kw=HYBRID.cap)

    RUN.committed = {}                       # replaced by the joint result
    if res["status"] != "optimal":
        msg = ("no schedule satisfies the constraints: " +
               "; ".join(f"{t.name} start>={_slot_time(t.earliest_start)}"
                         + (f", finish<={_slot_time(t.latest_finish)}"
                            if t.latest_finish is not None else "")
                         for t in tasks) +
               (f"; power cap {HYBRID.cap} kW" if HYBRID.cap else ""))
        RUN.infeasibility_report = f"optimizer: {msg}"
        return {"status": "infeasible", "committed": False, "message": msg}

    out = []
    for t in tasks:
        s = res["starts"][t.name]
        RUN.committed[t.name] = {"slot": s, "date": d}
        end = s + t.slots
        out.append({"appliance": t.name, "start_time": _slot_time(s),
                    "end_time": _slot_time(end) if end < T else "24:00"})
    return {"status": "optimal", "committed": True, "date": d,
            "schedule": out,
            "estimated_net_cost_gbp": round(res["objective"], 4)}


HYBRID_TOOLKIT = [optimize_schedule, report_infeasibility]

SYSTEM_HYBRID = """You are the EcoHome Energy Advisor, an agent that schedules a UK \
household's flexible appliances on the Octopus Agile half-hourly tariff to \
minimise electricity cost.

Appliances you can schedule (fixed run lengths):
- washing_machine: 2.0 kW, 2 h
- dishwasher: 1.8 kW, 1.5 h
- ev_charger: 7.4 kW, 6 h

Time convention: all times refer to the scheduled day, 00:00 to 24:00. A \
run cannot start before 00:00 of that day.

You do not choose start times yourself. A deterministic optimizer computes \
the minimum-cost schedule from the day's prices (and the home's solar \
forecast where available). Your job is to translate the user's request into \
the optimizer's inputs.

Workflow for scheduling requests:
1. Read the request and any calendar or context information. For each \
requested appliance decide the earliest time it may start (not_before) and \
the time by which it must have finished (finish_by), if any, and whether a \
household power cap applies.
2. Call optimize_schedule with every requested appliance and all \
constraints (one call is simplest). It commits the schedule. Schedules you \
only describe in text are NOT executed.
3. If the optimizer reports that no feasible schedule exists, tell the user \
which requirement cannot be met. Do not relax or drop the user's \
constraints to obtain a schedule.
4. Reply with a concise summary: each appliance's start-end time and the \
estimated cost in GBP.

Rules:
- Resolve every user constraint before calling the optimizer. A deadline of \
HH:MM means finished strictly by HH:MM.
- If the user's requirements cannot all be satisfied, use \
report_infeasibility (or rely on the optimizer's infeasibility result) \
instead of committing a schedule that breaks them.""" + _GUIDANCE


# ------------------------------------------------------------------ grids
def _eval_date(day: str) -> str:
    from datetime import date as _d, timedelta
    return (_d.fromisoformat(day) - timedelta(days=1)).isoformat()


EXP3_QUESTION = ("Please schedule my washing machine, dishwasher and EV charging for "
                 "tomorrow to minimise my electricity cost, making the most of my "
                 "solar generation, and schedule all three. The washing machine and "
                 "dishwasher can run any time; I work from home and only need the EV "
                 "ready by 18:00.")
PILOT_EXP2 = ["S1a", "S2a", "S3b", "S4a", "S5", "S6"]


def exp2_hybrid_specs(pilot: bool = False) -> List[Dict]:
    from experiments.grids import MODELS_EXP12, ALL_THREE
    from experiments.runner import run_id
    scen_dir = config.ARCHIVE / "scenarios"
    manifest = json.loads((scen_dir / "manifest.json").read_text(encoding="utf-8"))
    specs = []
    for model in MODELS_EXP12:
        for entry in manifest:
            if pilot and entry["id"] not in PILOT_EXP2:
                continue
            s = json.loads((scen_dir / f"{entry['id']}.json").read_text(encoding="utf-8"))
            for rep in range(1 if pilot else config.REPS_EXP2):
                rid = run_id(exp="exp2-hybrid", scenario=s["id"], model=model,
                             interface="fc", day=s["day"], rep=rep,
                             pv=PROMPT_VERSION_HYBRID)
                specs.append({
                    "exp": "exp2-hybrid", "scenario": s["id"], "model": model,
                    "interface": "fc", "day": s["day"], "rep": rep,
                    "eval_date": _eval_date(s["day"]),
                    "system_prompt": SYSTEM_HYBRID,
                    "prompt_version": PROMPT_VERSION_HYBRID,
                    "context": context_for(_eval_date(s["day"]), s.get("context_extras")),
                    "question": s["question"], "toolkit": HYBRID_TOOLKIT,
                    "requested": ALL_THREE, "constraints": s["constraints"],
                    "expected_infeasible": s["expected"].get("infeasible", False),
                    "inject_price_failure":
                        s["constraints"].get("inject_price_tool_failure", False),
                    "hybrid_use_pv": False,
                    "score": True, "run_id": rid})
    return specs


def exp3_hybrid_specs(pilot: bool = False) -> List[Dict]:
    from experiments.grids import MODELS3, ALL_THREE
    from experiments.runner import run_id
    from experiments.compute_baselines import EV_WFH_LATEST_FINISH
    sel = archive.load_selection()
    regimes = ("sunny", "mixed", "overcast")
    days = ([sel["exp3_regimes"][r][0] for r in regimes] if pilot else
            [d for r in regimes for d in sel["exp3_regimes"][r]])
    specs = []
    for model in MODELS3:
        for day in days:
            for rep in range(1 if pilot else config.REPS_EXP3):
                rid = run_id(exp="exp3-hybrid", scenario="hybrid", model=model,
                             interface="fc", day=day, rep=rep,
                             pv=PROMPT_VERSION_HYBRID)
                specs.append({
                    "exp": "exp3-hybrid", "scenario": "hybrid", "model": model,
                    "interface": "fc", "day": day, "rep": rep,
                    "eval_date": _eval_date(day),
                    "system_prompt": SYSTEM_HYBRID,
                    "prompt_version": PROMPT_VERSION_HYBRID,
                    "context": context_for(_eval_date(day)),
                    "question": EXP3_QUESTION, "toolkit": HYBRID_TOOLKIT,
                    "requested": ALL_THREE,
                    "constraints": {"ev_latest_finish": EV_WFH_LATEST_FINISH},
                    "use_pv_objective": True, "hybrid_use_pv": True,
                    "score": True, "run_id": rid})
    return specs


def exp2_drift_specs(pilot: bool = False) -> List[Dict]:
    """Optional provider-drift check: the DIRECT agent with the guided prompt,
    repeat 0 of every Exp 2 scenario and model, rerun at the time of the hybrid
    runs. Comparing with the July repeat-0 outcomes shows whether the models
    behind the same identifiers changed between the direct and hybrid runs.
    Identical specs to experiments.grids.exp2_specs except exp/run_id."""
    from experiments.grids import exp2_specs
    from experiments.runner import run_id
    out = []
    for s in exp2_specs():
        if s["prompt_version"] != "v2-guided" or s["rep"] != 0:
            continue
        if pilot and s["scenario"] not in PILOT_EXP2:
            continue
        s = dict(s)
        s["exp"] = "exp2-drift"
        s["run_id"] = run_id(exp="exp2-drift", scenario=s["scenario"], model=s["model"],
                             interface="fc", day=s["day"], rep=0, pv=s["prompt_version"])
        s["hybrid_use_pv"] = False
        out.append(s)
    return out


GRIDS = {"exp2-hybrid": exp2_hybrid_specs, "exp3-hybrid": exp3_hybrid_specs,
         "exp2-drift": exp2_drift_specs}
