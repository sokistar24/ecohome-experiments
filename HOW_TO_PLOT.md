# Generating the Paper Figures

## Setup (one time)
```
pip install matplotlib numpy
```

## Run from your project folder
All scripts run from `ecohome_experiments\` (same place as `agent.py`).

```
python plot_fig2_pareto.py       → figs/fig2_pareto.pdf + .png
python plot_fig3_taxonomy.py     → figs/fig3_taxonomy.pdf + .png
python plot_fig4_sensitivity.py  → figs/fig4_sensitivity.pdf + .png
python plot_fig5_cumulative.py   → figs/fig5_cumulative.pdf + .png
python plot_fig_money.py         → figs/fig_money.pdf + .png
```

## What each reads

| Script | CSV source |
|---|---|
| fig2_pareto | data/results/exp1_main.csv |
| fig3_taxonomy | data/results/exp2_main.csv |
| fig4_sensitivity | data/results/exp3_noise.csv + exp3_main.csv |
| fig5_cumulative | data/results/exp4_main.csv |
| fig_money | data/archive/prices/ + data/archive/weather_fc/ + day_selection.json |

## Adding figures to Overleaf / LaTeX
1. Create a `figs/` folder inside your Overleaf project (or the local folder next to `main.tex`)
2. Upload the `.pdf` files (PDFs give the best quality in IEEE format)
3. In `main.tex`, replace each `\fbox{...}` placeholder with:
   ```latex
   \includegraphics[width=\columnwidth]{figs/fig2_pareto}
   ```
   (for single-column figures) or
   ```latex
   \includegraphics[width=\textwidth]{figs/fig2_pareto}
   ```
   (for two-column / `figure*` figures)

## Which figure goes where in main.tex

| Figure | LaTeX label | Replace the \fbox in |
|---|---|---|
| fig2_pareto | fig:pareto | Exp 1 results section |
| fig3_taxonomy | fig:taxonomy | Exp 2 results section |
| fig4_sensitivity | fig:sensitivity | Exp 3 results section |
| fig_money | (illustrative day) | after fig4 in Exp 3 |
| fig5_cumulative | fig:cumulative | Exp 4 results section |
