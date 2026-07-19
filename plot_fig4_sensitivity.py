"""
Figure 4 --- Forecast-noise sensitivity of weather-aware scheduling.
Run from ecohome_experiments folder:
    python plot_fig4_sensitivity.py
Reads: data/results/exp3_noise.csv, data/results/exp3_main.csv
Output: figs/fig4_sensitivity.pdf + .png

Signed x-axis: accurate forecast at 0 (mean weather-aware cost), under-
forecast (negative noise) to the left, over-forecast (positive) to the
right. Every perturbed point lies above the accurate baseline, as it
must; the right (over-forecast) arm rises faster than the left, and the
price-only mean is drawn as a reference.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

# --- references from the main exp3 table (mean across regimes) -----------
main = list(csv.DictReader(open("data/results/exp3_main.csv")))
def mean_of(obj):
    vals = [float(r["mean_net_cost_realized"]) for r in main
            if r["objective"] == obj]
    return sum(vals) / len(vals)
accurate_wa = mean_of("weather_aware")
price_only  = mean_of("price_only")

# --- noise sweep ---------------------------------------------------------
noise = list(csv.DictReader(open("data/results/exp3_noise.csv")))
def ncost(tag):
    for r in noise:
        if r["noise"] == tag:
            return float(r["mean_net_cost_realized"])
    return None

under = {L: ncost(f"noise_{L}_m") for L in (10, 25, 50)}
over  = {L: ncost(f"noise_{L}_p") for L in (10, 25, 50)}

xs = [-50, -25, -10, 0, 10, 25, 50]
ys = [under[50], under[25], under[10], accurate_wa,
      over[10], over[25], over[50]]

fig, ax = plt.subplots(figsize=(6.0, 4.0))

ax.plot(xs, ys, "-", color="#333333", lw=2.5, alpha=0.45, zorder=2)
ax.scatter([-50, -25, -10], [under[50], under[25], under[10]],
           color="#2166ac", s=90, zorder=3, label="Under-forecast (\u2212)")
ax.scatter([10, 25, 50], [over[10], over[25], over[50]],
           color="#d73027", s=90, zorder=3, label="Over-forecast (+)")
ax.scatter([0], [accurate_wa], color="black", s=120, marker="D",
           zorder=4, label="Accurate forecast")

ax.axhline(price_only, color="grey", lw=2.5, ls=":", zorder=1)
ax.text(50, price_only, "  price-only mean", fontsize=14,
        ha="right", va="bottom", color="grey")

ax.set_xlabel("Forecast error (%)", fontsize=14)

# Define y-axis limits before using ymin in the text labels
ymin = min(ys + [price_only]) - 0.03
ymax = max(ys) + 0.03
ax.set_ylim(ymin, ymax)

ax.text(-50, ymin + 0.004, r"$\leftarrow$ under", fontsize=14,
        ha="left", color="#2166ac")
ax.text(50,  ymin + 0.004, r"over $\rightarrow$", fontsize=14,
        ha="right", color="#d73027")
ax.set_ylabel("Realized net cost (GBP/day)", fontsize=14)
ax.set_xticks(xs)
ax.set_xticklabels(["\u221250", "\u221225", "\u221210", "0",
                    "+10", "+25", "+50"], fontsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.grid(True, lw=0.5, alpha=0.35)
ax.legend(fontsize=14, loc="upper left", framealpha=0.95)

fig.tight_layout()
fig.savefig("figs/fig4_sensitivity.pdf", bbox_inches="tight")
fig.savefig("figs/fig4_sensitivity.png", dpi=180, bbox_inches="tight")
print("Saved figs/fig4_sensitivity.pdf and .png")