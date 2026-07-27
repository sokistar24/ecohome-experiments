"""Paired fc-vs-text cost gap on matched days, per model (rep-averaged).

Rows carry no explicit 'rep' field, so we aggregate the (up to 3) reps
within each (model, interface, day) into a mean gap over that cell's
SUCCESSFUL scored runs, then pair fc-mean vs text-mean on days where BOTH
interfaces have at least one successful run. This removes the denominator
mismatch behind the earlier 'text looks better for open models' effect.
"""
import json
import statistics as st
from collections import defaultdict

rows = [json.loads(l) for l in open("data/runs/exp1.jsonl") if l.strip()]
rows = [r for r in rows if r.get("scenario") == "multi" and r.get("score")]

# (model, interface, day) -> list of gaps from SUCCESSFUL scored runs
cell = defaultdict(list)
for r in rows:
    s = r["score"]
    if s.get("success") and s.get("cost_gap") is not None:
        cell[(r["model"], r["interface"], r["day"])].append(s["cost_gap"])

def cell_mean(model, iface, day):
    v = cell.get((model, iface, day))
    return st.mean(v) if v else None

models = sorted({m for (m, _, _) in cell})
days = sorted({d for (_, _, d) in cell})

print(f"{'model':12}{'n_days':>8}{'fc_gap%':>10}{'text_gap%':>11}{'text-fc':>10}")
print("-" * 51)
for model in models:
    pairs = []
    for d in days:
        fc = cell_mean(model, "fc", d)
        tx = cell_mean(model, "text", d)
        if fc is not None and tx is not None:      # both succeeded on day d
            pairs.append((fc, tx))
    if not pairs:
        print(f"{model:12}{0:>8}   (no days where both interfaces succeeded)")
        continue
    fc_m = 100 * st.mean(p[0] for p in pairs)
    tx_m = 100 * st.mean(p[1] for p in pairs)
    print(f"{model:12}{len(pairs):>8}{fc_m:>10.3f}{tx_m:>11.3f}{tx_m-fc_m:>10.3f}")

print("\nMatched days only (both interfaces produced >=1 valid schedule):")
print("  text-fc > 0 : text genuinely worse (expected).")
print("  text-fc ~ 0 : comparable on shared days => earlier 'text better' was")
print("                survivorship (text dropped hard days from its average).")
print("  text-fc < 0 : text really lower even matched => a real effect.")
