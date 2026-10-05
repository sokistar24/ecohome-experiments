"""
Phase C, step 0 -- build every table and figure used in the revised paper.

Reruns every analysis (A1-A6, B3, C1, C2) so all outputs are current and
consistent with the latest code (the number audit, c4, reads them),
then writes paper-ready assets to revision/outputs/paper/:

  tab_exp1.tex           Table 4  task completion and action interface (Exp 1)
  tab_exp2_arms.tex      Table 5  direct / guard / hybrid under constraint conflict
  tab_hybrid_fail.tex    Table 6  sources of hybrid failures
  tab_exp3.tex           Table 7  weather value: agents, hybrid and MILP references
  tab_exp4a.tex          Table 8  seven-day deployment
  tab_exp2_prompts.tex   Appendix  direct agent, baseline vs guided prompt, with severity
  fig_efficiency.pdf     Fig. 2   correct rate vs cost per correct schedule, direct -> hybrid
  fig_direct_vs_hybrid.pdf  Fig. 3  failure rate by scenario family, direct vs hybrid
  fig_sunny_day.pdf      Fig. 4   one sunny day: schedules of each arm against price and PV
  assets_preview.tex     a stand-alone document that shows all of the above

Tables are LaTeX `tabular` blocks (booktabs, makecell); captions and labels
stay in article.tex. Every number is read from the run logs or the Phase A/B
outputs; nothing is typed by hand.

No LLM calls. Run from the repo root (after Phase B):
    python -m revision.c0_paper_assets
"""
from __future__ import annotations

import collections
import contextlib
import csv
import io
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.patches as mpatches               # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

from experiments import archive, config             # noqa: E402
from experiments.optimizer import ApplianceTask, evaluate_schedule  # noqa: E402

ROOT = config.ROOT
OUTS = ROOT / "revision" / "outputs"
PAPER = OUTS / "paper"
PAPER.mkdir(parents=True, exist_ok=True)
RUNS = config.RUNS
F = config.EXPORT_RATE_GBP

MODELS5 = ["gpt", "gemini", "claude", "llama-3.3", "qwen-3"]
MODELS3 = ["gpt", "gemini", "claude"]
LAB = {"gpt": "GPT-4o-mini", "gemini": "Gemini 2.5 Flash", "claude": "Claude Sonnet 4.6",
       "llama-3.3": "Llama-3.3 70B", "qwen-3": "Qwen-3 32B"}
COL = {"gpt": "#2166ac", "gemini": "#d73027", "claude": "#4dac26",
       "llama-3.3": "#8073ac", "qwen-3": "#e08214"}
OPEN = {"llama-3.3", "qwen-3"}
APP_COL = {"ev_charger": "#2166ac", "washing_machine": "#4dac26", "dishwasher": "#d73027"}
APP_SHORT = {"ev_charger": "EV", "washing_machine": "WM", "dishwasher": "DW"}
SUBLANE = {"washing_machine": 0.28, "ev_charger": 0.0, "dishwasher": -0.28}   # within a lane


def rows(name):
    return list(csv.DictReader(open(OUTS / name, encoding="utf-8")))


def write(name, text):
    (PAPER / name).write_text(text, encoding="utf-8")
    print(f"  wrote revision/outputs/paper/{name}")


def pct(x):
    return f"{100 * float(x):.0f}"


def ci(s):
    """'[45, 89]' -> '[45, 89]' with a thin space style for LaTeX."""
    return s.replace(", ", ",\\,")


# ------------------------------------------------------------------ rerun
def rerun_analyses():
    from revision import (a1_rescore_departure_buffer as a1, a2_exp3_milp_reference as a2,
                          a3_exp4a_weather_value as a3, a4_exp2_violation_severity as a4,
                          a5_metrics_and_intervals as a5, a6_weekly_planning_check as a6,
                          b3_hybrid_analysis as b3, c1_coupling_check as c1,
                          c2_drift_check as c2)
    for name, mod in (("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4), ("A5", a5), ("A6", a6),
                      ("B3", b3), ("C1", c1), ("C2", c2)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main()
        print(f"  {name} rerun OK")


# ------------------------------------------------------------------ tables
def tab_exp1():
    a5 = {(r["model"], r["interface"]): r for r in rows("a5_table4_extended.csv")}
    pub = {(r["model"], r["interface"]): r
           for r in csv.DictReader(open(config.DATA / "results" / "exp1_main.csv", encoding="utf-8"))}
    out = [r"\begin{tabular}{llcccccrr}", r"\toprule",
           r"Model & Interface & \makecell{Success\\(\%)} & \makecell{Exact optimal\\(\%)} & "
           r"\makecell{Near-opt.\\(\%)} & \makecell{$\Delta J$ median / max\\(GBP)} & "
           r"\makecell{Max $\gamma$\\(\%)} & Tokens & \makecell{\$ per\\success} \\", r"\midrule"]
    for i, m in enumerate(MODELS5):
        if i == 3:
            out.append(r"\midrule")
        for itf in ("fc", "text"):
            r, p = a5[(m, itf)], pub[(m, itf)]
            name = LAB[m] if itf == "fc" else ""
            out.append(f"{name} & {'FC' if itf == 'fc' else 'Text'} & "
                       f"{pct(r['success'])} {ci(r['success_ci'])} & "
                       f"{pct(r['optimal'])} {ci(r['optimal_ci'])} & "
                       f"{pct(r['near_opt_successful_only'])} & "
                       f"{float(r['median_gap_gbp']):.2f} / {float(r['max_gap_gbp']):.2f} & "
                       f"{float(r['max_gap_pct']):.1f} & "
                       f"{float(p['mean_tokens']):,.0f} & {float(r['usd_per_success']):.4f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp1.tex", "\n".join(out) + "\n")


def invalid_commits():
    """Invalid commits on feasible scenarios, direct-guided vs hybrid (A1 rule)."""
    import copy
    from revision.a1_rescore_departure_buffer import load_scenarios, detect_buffered, family
    from revision.a4_exp2_violation_severity import measure, starts_of
    scen = load_scenarios()
    buf = detect_buffered(scen)
    res = collections.Counter()
    for name, keep in (("exp2", lambda r: r["prompt_version"] == "v2-guided"),
                       ("exp2-hybrid", lambda r: True)):
        arm = "direct" if name == "exp2" else "hybrid"
        for l in open(RUNS / f"{name}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if not keep(r) or family(r["scenario"]) == "S4":
                continue
            cons = copy.deepcopy(scen[r["scenario"]]["constraints"])
            if r["scenario"] in buf:
                cons["ev_latest_finish"] = buf[r["scenario"]]
            if any(v > 0 for v in measure(starts_of(r), cons).values()):
                res[(r["model"], arm)] += 1
    return res


def tab_exp2_arms():
    arms = {r["model"]: r for r in rows("b3_exp2_arms.csv")}
    inv = invalid_commits()
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Model & \makecell{Direct\\(\%)} & \makecell{Guard\\(\%)} & \makecell{Hybrid\\(\%)} & "
           r"\makecell{Hybrid $-$ direct\\(pp)} & \makecell{Invalid commits\\direct $\to$ hybrid} & "
           r"\makecell{\$ per correct\\direct / hybrid} \\", r"\midrule"]
    for i, m in enumerate(MODELS5):
        if i == 3:
            out.append(r"\midrule")
        r = arms[m]
        out.append(f"{LAB[m]} & {pct(r['direct_rate'])} {ci(r['direct_ci'])} & "
                   f"{pct(r['guard_rate'])} {ci(r['guard_ci'])} & "
                   f"{pct(r['hybrid_rate'])} {ci(r['hybrid_ci'])} & "
                   f"{float(r['hybrid_minus_direct_pp']):+.0f} {ci(r['diff_ci'])} & "
                   f"{inv[(m, 'direct')]} $\\to$ {inv[(m, 'hybrid')]} & "
                   f"{float(r['direct_usd_per_correct']):.4f} / {float(r['hybrid_usd_per_correct']):.4f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp2_arms.tex", "\n".join(out) + "\n")
    return arms


CAUSE_LABEL = [
    ("missed or loosened the EV deadline", "Interpretation",
     "Dropped or loosened the EV deadline $\\Rightarrow$ invalid commit"),
    ("false infeasibility (power cap)", "Interpretation",
     "Invented a power cap $\\Rightarrow$ false infeasibility"),
    ("false infeasibility (ev_charger earliest start)", "Interpretation",
     "Invented an EV earliest start $\\Rightarrow$ false infeasibility"),
    ("false infeasibility (washing_machine deadline)", "Interpretation",
     "Invented a washing-machine deadline $\\Rightarrow$ false infeasibility"),
    ("false infeasibility (ev_charger earliest start, power cap", "Interpretation",
     "Invented several constraints $\\Rightarrow$ false infeasibility"),
    ("tool failure not retried", "Protocol", "Did not retry after the injected tool failure"),
    ("no optimizer call", "Protocol", "No optimizer call and no infeasibility report"),
]


def tab_hybrid_fail():
    fails = rows("b3_hybrid_failures.csv")
    groups = collections.OrderedDict()
    for key, kind, label in CAUSE_LABEL:
        groups[(kind, label)] = [f for f in fails if key in f["cause"]
                                 and (key != "false infeasibility (power cap)"
                                      or f["cause"].endswith("(power cap)"))]
    assigned = sum(len(v) for v in groups.values())
    assert assigned == len(fails), f"{len(fails) - assigned} hybrid failures not mapped to a row"
    out = [r"\begin{tabular}{llrll}", r"\toprule",
           r"Type & Cause & Runs & Models & Scenarios \\", r"\midrule"]
    last = None
    for (kind, label), fs in groups.items():
        if not fs:
            continue
        if last and kind != last:
            out.append(r"\midrule")
        who = collections.Counter(f["model"] for f in fs)
        sc = sorted({f["scenario"] for f in fs})
        out.append(f"{kind if kind != last else ''} & {label} & {len(fs)} & "
                   + ", ".join(f"{LAB[m].split()[0]} {n}" for m, n in sorted(who.items())) + " & "
                   + ", ".join(sc) + r" \\")
        last = kind
    n_int = sum(len(v) for (k, _), v in groups.items() if k == "Interpretation")
    out += [r"\midrule", f" & Total & {len(fails)} & \\multicolumn{{2}}{{l}}{{{n_int} interpretation, "
            f"{len(fails) - n_int} protocol}} \\\\", r"\bottomrule", r"\end{tabular}"]
    write("tab_hybrid_fail.tex", "\n".join(out) + "\n")


def tab_exp3():
    b3 = {r["regime"]: r for r in rows("b3_exp3.csv")}
    a2 = {(r["regime"], r["arm"]): r for r in rows("a2_table6_extended.csv")}
    out = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{LLM agents (matched model-days)} & "
           r"\multicolumn{3}{c}{MILP references} & \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Regime & \makecell{Direct,\\price-only} & \makecell{Direct,\\weather tools} & Hybrid & "
           r"\makecell{Price-\\only} & \makecell{Weather-\\aware} & \makecell{Perfect\\foresight} & "
           r"\makecell{Hybrid captures\\(\%)} \\", r"\midrule"]
    for reg in ("overcast", "mixed", "sunny"):
        r = b3[reg]
        pf = float(a2[(reg, "Perfect-foresight MILP")]["net_cost"])
        out.append(f"{reg.capitalize()} & {float(r['price_only_agent']):.3f} & "
                   f"{float(r['weather_aware_agent']):.3f} & {float(r['hybrid']):.3f} & "
                   f"{float(r['price_only_milp']):.3f} & {float(r['weather_aware_milp']):.3f} & "
                   f"{pf:.3f} & {float(r['hybrid_weather_value_captured_pct']):.0f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp3.tex", "\n".join(out) + "\n")


def tab_exp4a():
    t = rows("a3_table7.csv")
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"Policy & \makecell{Seven-day cost\\(GBP)} & \makecell{EV deadline\\met (days)} & "
           r"\makecell{Timer-to-oracle\\savings captured (\%)} & "
           r"\makecell{Weather value\\captured (\%)} \\", r"\midrule"]
    for i, r in enumerate(t):
        if r["policy"] in ("Price-only MILP", "GPT-4o-mini agent"):
            out.append(r"\midrule")
        wv = r["weather_value_captured_pct"]
        wv = f"{float(wv):.1f}" if wv not in ("", None) else "--"
        name = r["policy"].replace("Weather-aware MILP (oracle)", "Weather-aware MILP (oracle)")
        out.append(f"{name} & {float(r['week_cost_gbp']) + 1e-9:.2f} & {r['ev_deadline_met_days'].split('/')[0]}/7 & "
                   f"{float(r['timer_to_oracle_share_pct']):.1f} & {wv} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp4a.tex", "\n".join(out) + "\n")


def tab_exp2_prompts():
    fam_key = {"deadline_conflict": "S1", "power_cap": "S2", "irregular_calendar": "S3",
               "infeasible": "S4", "instruction_vs_calendar": "S5", "tool_failure": "S6"}
    rates = collections.defaultdict(dict)
    for r in rows("a1_exp2_main_rescored.csv"):
        v = r["success_rate"] or r["infeasibility_reported_rate"]
        rates[(r["model"], fam_key[r["family"]])][r["prompt"]] = float(v)
    sev = {r["model"]: r for r in rows("a4_severity_by_model.csv") if r["prompt"] == "both"}
    out = [r"\begin{tabular}{l" + "c" * 6 + "ccc}", r"\toprule",
           r"Model & S1 & S2 & S3 & S4 & S5 & S6 & \makecell{Invalid\\commits} & "
           r"\makecell{Worst\\overrun (h)} & \makecell{Worst cap\\excess (kW)} \\", r"\midrule"]
    for i, m in enumerate(MODELS5):
        if i == 3:
            out.append(r"\midrule")
        cells = [f"{rates[(m, f)]['v1']:.2f}/{rates[(m, f)]['v2-guided']:.2f}"
                 for f in ("S1", "S2", "S3", "S4", "S5", "S6")]
        s = sev[m]
        late = f"{float(s['max_overrun_h']):.1f}" if int(s["deadline_violations"]) else "--"
        cap = f"{float(s['max_cap_excess_kw']):.1f}" if int(s["cap_violations"]) else "--"
        out.append(f"{LAB[m]} & " + " & ".join(cells)
                   + f" & {s['invalid_commits']}/{s['committing_runs']} & {late} & {cap} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp2_prompts.tex", "\n".join(out) + "\n")


# ------------------------------------------------------------------ figures
def fig_efficiency(arms):
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for m in MODELS5:
        r = arms[m]
        x0, y0 = float(r["direct_usd_per_correct"]), 100 * float(r["direct_rate"])
        x1, y1 = float(r["hybrid_usd_per_correct"]), 100 * float(r["hybrid_rate"])
        mk = "s" if m in OPEN else "o"
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=COL[m], lw=1.6,
                                    shrinkA=7, shrinkB=7, alpha=0.85))
        ax.scatter([x0], [y0], s=90, marker=mk, facecolors="white", edgecolors=COL[m],
                   linewidths=2.0, zorder=3)
        ax.scatter([x1], [y1], s=90, marker=mk, color=COL[m], edgecolors="white",
                   linewidths=1.0, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Inference cost per correct schedule (USD, log scale)", fontsize=10)
    ax.set_ylabel("Correct behaviour (%)", fontsize=10)
    ax.set_ylim(30, 104)
    ax.grid(lw=0.3, alpha=0.5, which="both")
    handles = [Line2D([0], [0], marker=("s" if m in OPEN else "o"), color="w",
                      markerfacecolor=COL[m], markersize=9, label=LAB[m]) for m in MODELS5]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                       markeredgecolor="0.3", markersize=9, label="Direct (open marker)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="0.3",
                       markersize=9, label="Hybrid (filled marker)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.95, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(PAPER / "fig_efficiency.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "fig_efficiency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote revision/outputs/paper/fig_efficiency.pdf")


def fig_direct_vs_hybrid():
    cmd = [sys.executable, str(ROOT / "plot_fig3_taxonomy.py"),
           "--csv", str(OUTS / "b3_fig_direct_vs_hybrid.csv"),
           "--left-prompt", "direct", "--right-prompt", "hybrid",
           "--left-title", "(a) Direct scheduling",
           "--right-title", "(b) Hybrid: LLM interprets, MILP schedules",
           "--out", str(PAPER / "fig_direct_vs_hybrid")]
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    print("  wrote revision/outputs/paper/fig_direct_vs_hybrid.pdf")


def _modal(schedules):
    c = collections.Counter(tuple(sorted(s.items())) for s in schedules if s)
    if not c:
        return None, 0
    top, n = c.most_common(1)[0]
    return dict(top), n


def fig_sunny_day():
    sel = archive.load_selection()
    base = json.loads((config.DATA / "results" / "baselines.json").read_text(encoding="utf-8"))["per_day"]
    sunny = sel["exp3_regimes"]["sunny"]
    # selection rule: the sunny day on which the forecast is worth most to the MILP
    val = {d: base[d]["wfh_ev18"]["price_only_milp"]["net_cost_realized"]
           - base[d]["wfh_ev18"]["oracle"]["net_cost_realized"] for d in sunny}
    day = max(val, key=val.get)
    prices = list(archive.load_prices(day))
    pv_fc, pv_act = archive.pv_slots(day, "fc"), archive.pv_slots(day, "actual")
    tasks = [ApplianceTask(a, config.APPLIANCES[a]["power_kw"], config.APPLIANCES[a]["slots"])
             for a in ("washing_machine", "dishwasher", "ev_charger")]

    def realized(starts):
        return evaluate_schedule(starts, tasks, prices, export_rate=F, pv=pv_act)["net_cost"]

    direct = collections.defaultdict(list)
    hybrid = []
    for l in open(RUNS / "exp3.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["day"] == day and r["scenario"] == "weather_aware":
            direct[r["model"]].append((r["score"] or {}).get("committed_starts") or {})
    for l in open(RUNS / "exp3-hybrid.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["day"] == day:
            hybrid.append((r["score"] or {}).get("committed_starts") or {})
    lanes = [("Price-only MILP", base[day]["wfh_ev18"]["price_only_milp"]["starts"], "")]
    for m in MODELS3:
        s, n = _modal(direct[m])
        k = sum(1 for x in direct[m] if x)
        lanes.append((f"Direct, {LAB[m]}", s, f"{n}/{len(direct[m])} runs" if s else "no schedule"))
    hs, hn = _modal(hybrid)
    lanes.append(("Hybrid, all three models", hs, f"{hn}/{len(hybrid)} runs"))
    lanes.append(("Weather-aware MILP", base[day]["wfh_ev18"]["oracle"]["starts"], ""))

    T = config.SLOTS_PER_DAY
    slots = list(range(T))
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 2.1]})
    axes[0].step(slots + [T], [100 * p for p in prices] + [100 * prices[-1]], where="post",
                 color="black", lw=1.4)
    axes[0].axhline(0, color="grey", lw=0.5, ls="--")
    axes[0].set_ylabel("Import price\n(p/kWh)", fontsize=9)
    axes[0].grid(lw=0.3, alpha=0.4)
    axes[1].bar([s + 0.5 for s in slots], pv_fc, width=1.0, color="#fdae61", label="Forecast")
    axes[1].step(slots + [T], list(pv_act) + [pv_act[-1]], where="post", color="#b35806",
                 lw=1.2, label="Realized")
    axes[1].set_ylabel("PV\n(kWh/slot)", fontsize=9)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(lw=0.3, alpha=0.4)
    ax = axes[2]
    for yi, (label, starts, note) in enumerate(reversed(lanes)):
        if starts:
            for a, s in starts.items():
                d = config.APPLIANCES[a]["slots"]
                y = yi + SUBLANE[a]
                ax.barh(y, d, left=s, height=0.26, color=APP_COL[a], alpha=0.9,
                        edgecolor="white", linewidth=0.6)
                ax.text(s + d / 2, y, APP_SHORT[a], ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
            txt = f"\u00a3{realized(starts):.2f}" + (f"  ({note})" if note else "")
        else:
            txt = note
        ax.text(T + 0.5, yi, txt, va="center", fontsize=7.5)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([l for l, _, _ in reversed(lanes)], fontsize=8.5)
    ax.set_xlim(0, T)
    ax.set_xticks(range(0, T + 1, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=8)
    ax.set_xlabel("Time of day", fontsize=9)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    ax.legend(handles=[mpatches.Patch(color=APP_COL[a], label=APP_SHORT[a]) for a in APP_COL],
              fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3, frameon=False)
    fig.suptitle(f"{day} (sunny)", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(PAPER / "fig_sunny_day.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "fig_sunny_day.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote revision/outputs/paper/fig_sunny_day.pdf (day {day}, forecast worth "
          f"GBP {val[day]:.2f} to the MILP; lanes: "
          + "; ".join(f"{l} {n}" for l, _, n in lanes if n) + ")")
    return day


# ------------------------------------------------------------------ preview
PREVIEW = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,makecell,graphicx,amsmath}
\begin{document}
\section*{Step 0 asset preview}
\begin{table}[h]\centering\footnotesize
\caption{Task completion and action interface (Exp.~1).}
\resizebox{\linewidth}{!}{\input{tab_exp1.tex}}\end{table}
\begin{table}[h]\centering\footnotesize
\caption{Direct, guard and hybrid under constraint conflict (Exp.~2).}
\resizebox{\linewidth}{!}{\input{tab_exp2_arms.tex}}\end{table}
\begin{table}[h]\centering\footnotesize
\caption{Sources of hybrid failures.}
\resizebox{\linewidth}{!}{\input{tab_hybrid_fail.tex}}\end{table}
\begin{table}[h]\centering\footnotesize
\caption{Realized net cost (GBP/day) by weather regime (Exp.~3).}
\resizebox{\linewidth}{!}{\input{tab_exp3.tex}}\end{table}
\begin{table}[h]\centering\footnotesize
\caption{Seven-day deployment (Exp.~4a).}
\resizebox{\linewidth}{!}{\input{tab_exp4a.tex}}\end{table}
\begin{table}[h]\centering\footnotesize
\caption{Appendix: direct agent, baseline/guided prompt, with severity.}
\resizebox{\linewidth}{!}{\input{tab_exp2_prompts.tex}}\end{table}
\clearpage
\begin{figure}[h]\centering\includegraphics[width=0.75\linewidth]{fig_efficiency.pdf}
\caption{Correct behaviour vs cost per correct schedule, direct to hybrid.}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{fig_direct_vs_hybrid.pdf}
\caption{Failure rate by scenario family, direct vs hybrid.}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=0.9\linewidth]{fig_sunny_day.pdf}
\caption{Illustrative sunny day.}\end{figure}
\end{document}
"""


def main():
    print("C0 -- building paper assets")
    rerun_analyses()
    tab_exp1()
    arms = tab_exp2_arms()
    tab_hybrid_fail()
    tab_exp3()
    tab_exp4a()
    tab_exp2_prompts()
    fig_efficiency(arms)
    fig_direct_vs_hybrid()
    fig_sunny_day()
    write("assets_preview.tex", PREVIEW)
    print("Done. Compile revision/outputs/paper/assets_preview.tex to review every asset.")


if __name__ == "__main__":
    main()
