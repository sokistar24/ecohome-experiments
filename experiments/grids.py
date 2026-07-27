"""
Grid builders: turn day_selection.json + scenarios/ into the exact list of
run specs for each experiment. Kept separate from the runner so the grid
(and thus --dry-run cost estimates and run_ids) is pure and testable.

Every spec dict carries what execute() needs; run_id is derived from the
identity fields only (exp, scenario, model, interface, day, rep,
prompt_version) so reruns are idempotent and --force-free additions (e.g.
a new model) don't disturb existing ids.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Dict, List

from experiments import archive, config
from experiments.eval_prompts import (PROMPT_VERSION_BASELINE,
                                      PROMPT_VERSION_GUIDED,
                                      PROMPT_VERSION_PRICE_ONLY,
                                      SYSTEM_BASELINE, SYSTEM_GUIDED,
                                      SYSTEM_PRICE_ONLY, context_for)
from experiments.eval_tools import FULL_TOOLKIT, PRICE_ONLY_TOOLKIT
from experiments.runner import STANDARD_Q, SINGLE_Q, run_id

ALL_THREE = ["washing_machine", "dishwasher", "ev_charger"]
EV_LF = 15                                    # 07:30 finish
MODELS3 = ["gpt", "gemini", "claude"]
# Open-weight extension: these enter ONLY Exp 1 and Exp 2 (coordination and
# constraint-conflict), the axes where the open-vs-closed comparison to [17]
# is meaningful. Exp 3/4 (weather regime, longitudinal) stay closed-only by
# construction -- their grids iterate MODELS3, so no open-model spec is ever
# generated for them.
MODELS_OPEN = ["llama-3.3", "qwen-3"]
MODELS_EXP12 = MODELS3 + MODELS_OPEN


def _eval_date(day: str) -> str:
    return (date.fromisoformat(day) - timedelta(days=1)).isoformat()


# ------------------------------------------------------------------ exp1
def exp1_specs(pilot: bool = False) -> List[Dict]:
    sel = archive.load_selection()
    terciles = sel["exp1_terciles"]
    if pilot:
        days = [terciles[t][0] for t in ("low", "mid", "high")]
        reps, interfaces = 1, ["fc"]
    else:
        days = [d for t in ("low", "mid", "high") for d in terciles[t]]
        reps, interfaces = config.REPS_EXP1, list(config.INTERFACES)
    specs = []
    for model in MODELS_EXP12:
        for interface in interfaces:
            for day in days:
                for kind, q, requested, cons in (
                    ("single", SINGLE_Q, ["washing_machine"], {}),
                    ("multi", STANDARD_Q, ALL_THREE, {"ev_latest_finish": EV_LF})):
                    for rep in range(reps):
                        rid = run_id(exp="exp1", scenario=kind, model=model,
                                     interface=interface, day=day, rep=rep,
                                     pv=PROMPT_VERSION_PRICE_ONLY)
                        specs.append({
                            "exp": "exp1", "scenario": kind, "model": model,
                            "interface": interface, "day": day, "rep": rep,
                            "eval_date": _eval_date(day),
                            "system_prompt": SYSTEM_PRICE_ONLY,
                            "prompt_version": PROMPT_VERSION_PRICE_ONLY,
                            "context": context_for(_eval_date(day)),
                            "question": q, "toolkit": PRICE_ONLY_TOOLKIT,
                            "requested": requested, "constraints": cons,
                            "score": True, "run_id": rid})
    return specs


# ------------------------------------------------------------------ exp2
def exp2_specs() -> List[Dict]:
    scen_dir = config.ARCHIVE / "scenarios"
    manifest = json.loads((scen_dir / "manifest.json").read_text())
    specs = []
    for model in MODELS_EXP12:
        for entry in manifest:
            s = json.loads((scen_dir / f"{entry['id']}.json").read_text())
            requested = ALL_THREE                       # every scenario asks all 3
            for pv_name, sysp, sysc in (
                (PROMPT_VERSION_BASELINE, SYSTEM_BASELINE, "baseline"),
                (PROMPT_VERSION_GUIDED, SYSTEM_GUIDED, "guided")):
                for rep in range(config.REPS_EXP2):
                    rid = run_id(exp="exp2", scenario=s["id"], model=model,
                                 interface="fc", day=s["day"], rep=rep,
                                 pv=pv_name)
                    specs.append({
                        "exp": "exp2", "scenario": s["id"], "model": model,
                        "interface": "fc", "day": s["day"], "rep": rep,
                        "eval_date": _eval_date(s["day"]),
                        "system_prompt": sysp, "prompt_version": pv_name,
                        "context": context_for(_eval_date(s["day"]),
                                               s.get("context_extras")),
                        "question": s["question"], "toolkit": FULL_TOOLKIT,
                        "requested": requested,
                        "constraints": s["constraints"],
                        "expected_infeasible": s["expected"].get("infeasible",
                                                                 False),
                        "inject_price_failure":
                            s["constraints"].get("inject_price_tool_failure",
                                                 False),
                        "score": True, "run_id": rid})
    return specs


# ------------------------------------------------------------------ exp3
def exp3_specs() -> List[Dict]:
    sel = archive.load_selection()
    days = [d for r in ("sunny", "mixed", "overcast")
            for d in sel["exp3_regimes"][r]]
    # Exp 3 uses the WFH EV deadline (18:00) so solar can matter (baselines note)
    from experiments.compute_baselines import EV_WFH_LATEST_FINISH
    q = ("Please schedule my washing machine, dishwasher and EV charging for "
         "tomorrow to minimise my electricity cost, making the most of my "
         "solar generation, and schedule all three. The washing machine and "
         "dishwasher can run any time; I work from home and only need the EV "
         "ready by 18:00.")
    specs = []
    for model in MODELS3:
        for day in days:
            for arm, sysp, pv, toolkit in (
                ("weather_aware", SYSTEM_BASELINE, PROMPT_VERSION_BASELINE,
                 FULL_TOOLKIT),
                ("price_only", SYSTEM_PRICE_ONLY, PROMPT_VERSION_PRICE_ONLY,
                 PRICE_ONLY_TOOLKIT)):
                for rep in range(config.REPS_EXP3):
                    rid = run_id(exp="exp3", scenario=arm, model=model,
                                 interface="fc", day=day, rep=rep, pv=pv)
                    specs.append({
                        "exp": "exp3", "scenario": arm, "model": model,
                        "interface": "fc", "day": day, "rep": rep,
                        "eval_date": _eval_date(day),
                        "system_prompt": sysp, "prompt_version": pv,
                        "context": context_for(_eval_date(day)),
                        "question": q, "toolkit": toolkit,
                        "requested": ALL_THREE,
                        "constraints": {"ev_latest_finish": EV_WFH_LATEST_FINISH},
                        "use_pv_objective": True,     # score net cost w/ PV
                        "score": True, "run_id": rid})
    return specs


def exp3_noise_specs(champion: str) -> List[Dict]:
    sel = archive.load_selection()
    days = [d for r in ("sunny", "mixed", "overcast")
            for d in sel["exp3_regimes"][r]]
    from experiments.compute_baselines import EV_WFH_LATEST_FINISH
    q = ("Please schedule my washing machine, dishwasher and EV charging for "
         "tomorrow to minimise cost using my solar generation, and schedule "
         "all three. I only need the EV ready by 18:00.")
    specs = []
    for day in days:
        for level in config.NOISE_LEVELS:
            for sign in (+1, -1):
                for rep in range(config.REPS_EXP3):
                    tag = f"noise_{int(level*100)}_{'p' if sign>0 else 'm'}"
                    rid = run_id(exp="exp3-noise", scenario=tag,
                                 model=champion, interface="fc", day=day,
                                 rep=rep, pv=PROMPT_VERSION_BASELINE)
                    specs.append({
                        "exp": "exp3-noise", "scenario": tag,
                        "model": champion, "interface": "fc", "day": day,
                        "rep": rep, "eval_date": _eval_date(day),
                        "system_prompt": SYSTEM_BASELINE,
                        "prompt_version": PROMPT_VERSION_BASELINE,
                        "context": context_for(_eval_date(day)),
                        "question": q, "toolkit": FULL_TOOLKIT,
                        "requested": ALL_THREE,
                        "constraints": {"ev_latest_finish": EV_WFH_LATEST_FINISH},
                        "pv_noise": (level, sign), "use_pv_objective": True,
                        "score": True, "run_id": rid})
    return specs


# ------------------------------------------------------------------ exp4
def exp4a_specs() -> List[Dict]:
    """Week-long rolling deployment: 7 consecutive days, one schedule per
    simulated evening. Uses week 1 from the selection."""
    sel = archive.load_selection()
    week = sel["exp4_weeks"][0]
    q = ("Please schedule my washing machine, dishwasher and EV charging for "
         "tomorrow to minimise cost using my solar, and schedule all three. "
         "I only need the EV ready by 18:00.")
    from experiments.compute_baselines import EV_WFH_LATEST_FINISH
    specs = []
    for model in MODELS3:
        for pass_i in range(config.PASSES_EXP4A):
            for day in week:
                rid = run_id(exp="exp4a", scenario=f"pass{pass_i}",
                             model=model, interface="fc", day=day,
                             rep=pass_i, pv=PROMPT_VERSION_BASELINE)
                specs.append({
                    "exp": "exp4a", "scenario": f"pass{pass_i}",
                    "model": model, "interface": "fc", "day": day,
                    "rep": pass_i, "eval_date": _eval_date(day),
                    "system_prompt": SYSTEM_BASELINE,
                    "prompt_version": PROMPT_VERSION_BASELINE,
                    "context": context_for(_eval_date(day)),
                    "question": q, "toolkit": FULL_TOOLKIT,
                    "requested": ALL_THREE,
                    "constraints": {"ev_latest_finish": EV_WFH_LATEST_FINISH},
                    "use_pv_objective": True, "score": True, "run_id": rid})
    return specs


GRIDS = {"exp1": exp1_specs, "exp2": exp2_specs, "exp3": exp3_specs,
         "exp4a": exp4a_specs}


def exp4b_specs(champion: str) -> List[Dict]:
    """Weekly joint planning (champion only): 2 archived weeks x REPS_EXP4B.
    The agent must place 2 WM + 3 DW cycles across the week plus weekday EV
    charges; scored by analyze against the weekly MILP (Appendix A variant)."""
    sel = archive.load_selection()
    q = ("Plan my whole week to minimise electricity cost and schedule "
         "everything: run the washing machine twice and the dishwasher three "
         "times across the week (each on a different day), and charge the EV "
         "every weekday so it is ready by 08:00. Use the weekly prices and my "
         "solar. Commit each run with schedule_appliance.")
    specs = []
    for wi, week in enumerate(sel["exp4_weeks"][:config.EXP4B_WEEKS]):
        for rep in range(config.REPS_EXP4B):
            rid = run_id(exp="exp4b", scenario=f"week{wi}", model=champion,
                         interface="fc", day=week[0], rep=rep,
                         pv=PROMPT_VERSION_BASELINE)
            specs.append({
                "exp": "exp4b", "scenario": f"week{wi}", "model": champion,
                "interface": "fc", "day": week[0], "week": week, "rep": rep,
                "eval_date": _eval_date(week[0]),
                "system_prompt": SYSTEM_BASELINE,
                "prompt_version": PROMPT_VERSION_BASELINE,
                "context": context_for(_eval_date(week[0])),
                "question": q, "toolkit": FULL_TOOLKIT,
                "requested": [],            # weekly scoring handled in analyze
                "constraints": {}, "score": False, "run_id": rid})
    return specs
