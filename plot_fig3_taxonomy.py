"""
Figure 3 --- Constraint-conflict failure heatmap: baseline vs guided prompt.

Run from ecohome_experiments folder:
    python plot_fig3_taxonomy.py
Reads: data/results/exp2_main.csv
Output: figs/fig3_taxonomy.pdf + .png

Cells are failure rate: for S1/S2/S3/S5/S6 this is 1 - success_rate;
for S4 (infeasible-task family) it is 1 - infeasibility_reported_rate,
since correct behavior is to REPORT infeasibility rather than commit.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

os.makedirs("figs", exist_ok=True)

MODELS = ["gpt", "gemini", "claude"]
MODEL_LABELS = {"gpt": "GPT-4o-mini",
                "gemini": "Gemini 2.5 Flash",
                "claude": "Claude Sonnet 4.6"}

FAMILIES = [
    ("S1", "deadline_conflict",       "success_rate"),
    ("S2", "power_cap",               "success_rate"),
    ("S3", "irregular_calendar",      "success_rate"),
    ("S4", "infeasible",              "infeasibility_reported_rate"),
    ("S5", "instruction_vs_calendar", "success_rate"),
    ("S6", "tool_failure",            "success_rate"),
]

rows = list(csv.DictReader(open("data/results/exp2_main.csv")))

def cell(model, prompt, fam_key, metric):
    vals = []
    for r in rows:
        if r["model"] != model:      continue
        if r["prompt"] != prompt:    continue
        if r["family"] != fam_key:   continue
        v = r.get(metric, "")
        if v == "" or v is None:     continue
        try:                         vals.append(float(v))
        except ValueError:           continue
    if not vals:
        return 0.0
    return 1.0 - sum(vals) / len(vals)

BASE   = np.array([[cell(m, "v1",        fk, mk) for _, fk, mk in FAMILIES]
                   for m in MODELS])
GUIDED = np.array([[cell(m, "v2-guided", fk, mk) for _, fk, mk in FAMILIES]
                   for m in MODELS])

cmap = LinearSegmentedColormap.from_list(
    "wr", [(1,1,1), (1,0.85,0.75), (0.90,0.45,0.35), (0.75,0.10,0.10)])

fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.4),
                        gridspec_kw={"wspace": 0.06})

fam_short = [f[0] for f in FAMILIES]
model_labels_full = [MODEL_LABELS[m] for m in MODELS]

for k, (ax, arr) in enumerate([(axes[0], BASE), (axes[1], GUIDED)]):
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fam_short)))
    ax.set_xticklabels(fam_short, fontsize=9)
    if k == 0:
        ax.set_yticks(range(len(model_labels_full)))
        ax.set_yticklabels(model_labels_full, fontsize=8.5)
    else:
        ax.set_yticks([])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            color = "white" if v > 0.55 else "black"
            txt = f"{v:.2f}" if v > 0.005 else "\u00b7"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

# small panel labels above each
axes[0].set_title("(a) Baseline prompt", fontsize=9, pad=6, loc="center")
axes[1].set_title("(b) Guided prompt",   fontsize=9, pad=6, loc="center")

# shared colorbar
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.85)
cbar.set_label("Failure rate", fontsize=8.5)
cbar.ax.tick_params(labelsize=8)

# S-code key spanning both panels
key = ("S1 deadline  \u00b7  S2 power cap  \u00b7  S3 irregular calendar  "
       "\u00b7  S4 infeasible  \u00b7  S5 instruction conflict  "
       "\u00b7  S6 tool failure")
fig.text(0.5, -0.06, key, ha="center", fontsize=7.5, style="italic")

fig.savefig("figs/fig3_taxonomy.pdf", bbox_inches="tight")
fig.savefig("figs/fig3_taxonomy.png", dpi=180, bbox_inches="tight")
print("Saved figs/fig3_taxonomy.pdf and .png")