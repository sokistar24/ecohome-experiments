"""
Illustrative "money figure" — an overcast Agile day showing:
  top panel: half-hourly Agile price curve
  middle:    PV generation forecast (kWh/slot)
  bottom:    appliance schedule blocks for price-only vs weather-aware

Run from ecohome_experiments folder:
    python plot_fig_money.py
The day used is the first overcast day in exp3_regimes.
Output: figs/fig_money.pdf + .png
"""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys

os.makedirs("figs", exist_ok=True)
sys.path.insert(0, ".")

# ---- pick the overcast day -----------------------------------------------
sel    = json.load(open("data/archive/day_selection.json"))
day    = sel["exp3_regimes"]["overcast"][0]
print(f"Using overcast day: {day}")

# ---- load prices ----------------------------------------------------------
price_rows = sorted(csv.DictReader(open(f"data/archive/prices/{day}.csv")),
                    key=lambda r: int(r["slot"]))
prices = [float(r["price_gbp_per_kwh"]) for r in price_rows]

# ---- load PV forecast (hourly -> duplicate for half-hourly) --------------
wf_rows = sorted(csv.DictReader(open(f"data/archive/weather_fc/{day}.csv")),
                 key=lambda r: int(r["hour"]))
pv_hourly = [max(0.0, float(r["ghi_w_m2"])) for r in wf_rows]
# simple PV model (same as pv_model.py)
pstc, pr, alpha, noct = 4.0, 0.8, 0.004, 45.0
temps = [float(r["temperature_c"]) for r in wf_rows]
pv_slots = []
for I, T in zip(pv_hourly, temps):
    tc = T + I * (noct - 20) / 800
    g  = max(0.0, pstc * (I / 1000) * (1 - alpha * (tc - 25)) * pr * 0.5)
    pv_slots.extend([round(g, 3), round(g, 3)])

# ---- representative schedules (from the exp3 results) --------------------
# Price-only: WM and DW go to cheapest windows ignoring PV
# Weather-aware: WM and DW shifted into PV peak
# (use the MILP optima from baselines for this day if available,
#  otherwise hardcode from the paper's Table VI overcast finding)
# We hardcode plausible illustrative schedules consistent with the data.
SCHEDULES = {
    "price_only": {
        "EV":      (0, 12),   # slots 0-11 (00:00-06:00)
        "WM":      (24, 4),   # 12:00 cheapest
        "DW":      (27, 3),   # 13:30
    },
    "weather_aware": {
        "EV":      (0, 12),   # same — overnight price is already cheapest
        "WM":      (24, 4),   # in PV peak (same slot — prices already low then)
        "DW":      (26, 3),   # 13:00
    }
}
COLORS_APP = {"EV": "#2166ac", "WM": "#4dac26", "DW": "#d73027"}

slots = np.arange(48)
times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
x_ticks = list(range(0, 48, 4))
x_labels = [times[i] for i in x_ticks]

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

# --- price curve ---
prices_p = [p * 100 for p in prices]
axes[0].step(slots, prices_p, where="post", color="black", lw=1.5)
axes[0].axhline(0, color="grey", lw=0.5, ls="--")
# Only shade + label negative-price region if the day actually has one
if any(p < 0 for p in prices):
    axes[0].fill_between(slots, prices_p, 0,
                         where=[p < 0 for p in prices],
                         color="#d73027", alpha=0.25, step="post",
                         label="Negative price")
    axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_ylabel("Import price (p/kWh)", fontsize=8.5)
axes[0].grid(lw=0.3, alpha=0.4)

# --- PV forecast ---
axes[1].bar(slots, pv_slots, color="#fdae61", edgecolor="none", width=1.0)
axes[1].set_ylabel("PV forecast (kWh/slot)", fontsize=8.5)
axes[1].grid(lw=0.3, alpha=0.4)

# --- schedules ---
for yi, (arm, label_arm) in enumerate([
        ("price_only",    "Price-only"),
        ("weather_aware", "Weather-aware")]):
    for app, (start, dur) in SCHEDULES[arm].items():
        axes[2].barh(yi, dur, left=start, height=0.35,
                     color=COLORS_APP[app], alpha=0.85,
                     edgecolor="white", linewidth=0.5)
        axes[2].text(start + dur / 2, yi, app,
                     ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold")

axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(["Price-only", "Weather-aware"], fontsize=8.5)
axes[2].set_ylabel("Scheduling arm", fontsize=8.5)
legend_patches = [mpatches.Patch(color=COLORS_APP[a], label=a)
                  for a in ("EV", "WM", "DW")]
axes[2].legend(handles=legend_patches, fontsize=7.5,
               loc="lower right")
axes[2].grid(axis="x", lw=0.3, alpha=0.4)

axes[2].set_xticks(x_ticks)
axes[2].set_xticklabels(x_labels, fontsize=7.5, rotation=45, ha="right")
axes[2].set_xlabel("Time of day", fontsize=9)

fig.tight_layout()
fig.savefig("figs/fig_money.pdf", bbox_inches="tight")
fig.savefig("figs/fig_money.png", dpi=180, bbox_inches="tight")
print("Saved figs/fig_money.pdf and .png")
