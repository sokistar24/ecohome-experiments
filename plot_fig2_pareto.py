"""
Figure 2 — FC-only cost-latency-optimality Pareto (design C).
Run from ecohome_experiments folder:
    python plot_fig2_pareto.py
Output: figs/fig2_pareto.pdf + .png

Shows only the three function-calling agents (the deployable
configurations). Axes are zoomed to the FC region so the three models
separate clearly. Bubble area is proportional to $/successful schedule;
the value is printed under each bubble, so no size legend is needed
(also stated in the caption). The text-parsed interface results are in
Table IV.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.makedirs("figs", exist_ok=True)
rows = [r for r in csv.DictReader(open("data/results/exp1_main.csv"))
        if r["interface"] == "fc"]

_models = {r["model"] for r in rows}
_want = {"gpt", "gemini", "claude", "llama-3.3", "qwen-3"}
_missing = _want - _models
if _missing:
    raise SystemExit(f"ERROR: exp1_main.csv (fc) missing models: {sorted(_missing)}. "
                     f"Present: {sorted(_models)}")

COL = {"gpt": "#2166ac", "gemini": "#d73027", "claude": "#4dac26",
       "llama-3.3": "#8073ac", "qwen-3": "#e08214"}
LAB = {"gpt": "GPT-4o-mini", "gemini": "Gemini 2.5 Flash",
       "claude": "Claude Sonnet 4.6",
       "llama-3.3": "Llama-3.3 70B", "qwen-3": "Qwen-3 32B"}
ORDER = ["gpt", "gemini", "claude", "llama-3.3", "qwen-3"]
OPEN = {"llama-3.3", "qwen-3"}

def bubble(cost):
    return 320 + cost * 15000

fig, ax = plt.subplots(figsize=(6.0, 4.0))

pts = {r["model"]: r for r in rows}
for m in ORDER:
    r = pts[m]
    opt  = float(r["optimality_rate"])
    lat  = float(r["mean_latency_s"])
    cost = float(r["cost_per_success_usd"])
    marker = "s" if m in OPEN else "o"   # squares = open-source
    ax.scatter(lat, opt, s=bubble(cost), c=COL[m], edgecolors="white",
               linewidths=2.2, alpha=0.90, zorder=3, marker=marker)

ax.axhline(1.0, color="grey", lw=1.0, ls="--", zorder=1)
ax.set_xlabel("Mean latency per run (s)  " + r"$\rightarrow$" + " slower",
              fontsize=14)
ax.set_ylabel("Optimality rate  ",
              fontsize=14)
ax.set_ylim(0.30, 1.06)
ax.set_xlim(0, 32)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.tick_params(axis="both", labelsize=13)
ax.grid(True, lw=0.5, alpha=0.35)

model_leg = [Line2D([0], [0], marker=("s" if m in OPEN else "o"), color="w",
                    markerfacecolor=COL[m], markersize=13, label=LAB[m])
             for m in ORDER]
ax.legend(handles=model_leg, fontsize=12, loc="lower left",
          framealpha=0.95)

fig.tight_layout()
fig.savefig("figs/fig2_pareto.pdf", bbox_inches="tight")
fig.savefig("figs/fig2_pareto.png", dpi=180, bbox_inches="tight")
print("Saved figs/fig2_pareto.pdf and .png")