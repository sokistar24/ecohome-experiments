"""
Figure 5 --- Cumulative realized net cost over 7 deployment days.
Run from ecohome_experiments folder:
    python plot_fig5_cumulative.py
Reads: data/runs/exp4a.jsonl  (real per-day agent costs)
       data/results/baselines.json  (reference policies)
Output: figs/fig5_cumulative.pdf + .png

Shows what was actually run: one cumulative trajectory per policy over
the 7 archived deployment days. No projection panel --- 4-week figures
are quoted only as illustrative extrapolations in the text.
"""
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

rows = [json.loads(l) for l in open("data/runs/exp4a.jsonl") if l.strip()]
per = defaultdict(list)
for r in rows:
    sc = r.get("score") or {}
    if sc.get("net_cost_realized") is not None:
        per[(r["model"], r["day"])].append(sc["net_cost_realized"])
days = sorted({d for _, d in per})

def daily_mean(model):
    return [sum(per[(model, d)]) / len(per[(model, d)]) for d in days]

base = json.loads(open("data/results/baselines.json").read())["per_day"]
def ref_daily(policy):
    return [base[d]["wfh_ev18"][policy]["net_cost_realized"] for d in days]

def cumsum(seq):
    out, s = [], 0.0
    for x in seq:
        s += x; out.append(s)
    return out

x = list(range(1, len(days) + 1))
fig, ax = plt.subplots(figsize=(7.0, 5.0))

PLOTS = [
    ("immediate",       "Immediate start",        "#d73027", "-",  2.5),
    ("off_peak_timer",  "Off-peak timer",         "#f46d43", "--", 2.5),
    ("greedy_slot",     "Greedy cheapest slot",   "#fdae61", ":",  2.5),
    ("price_only_milp", "Price-only MILP",        "#abd9e9", "-",  2.5),
    ("oracle",          "Extended MILP oracle",   "#1a1a1a", "--", 2.5),
]
for key, label, color, ls, lw in PLOTS:
    y = cumsum(ref_daily(key))
    ax.plot(x, y, color=color, ls=ls, lw=lw, label=label)

AGENTS = [
    ("gpt",    "GPT-4o-mini agent",       "#2166ac"),
    ("gemini", "Gemini 2.5 Flash agent",  "#4dac26"),
    ("claude", "Claude Sonnet 4.6 agent", "#9970ab"),
]
for key, label, color in AGENTS:
    y = cumsum(daily_mean(key))
    ax.plot(x, y, color=color, ls="-", lw=2.5, label=label, alpha=0.95)

ax.set_xlabel("Deployment day", fontsize=14)
ax.set_ylabel("Cumulative realized net cost (GBP)", fontsize=14)
ax.set_xticks(x)
ax.set_xlim(1, 7)
ax.tick_params(axis="both", labelsize=14)
ax.grid(True, lw=0.5, alpha=0.35)
ax.legend(fontsize=14, loc="upper left", framealpha=0.95, ncol=1)

fig.tight_layout()
fig.savefig("figs/fig5_cumulative.pdf", bbox_inches="tight")
fig.savefig("figs/fig5_cumulative.png", dpi=180, bbox_inches="tight")
print("Saved figs/fig5_cumulative.pdf and .png")
