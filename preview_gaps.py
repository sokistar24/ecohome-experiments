"""Preview: old (all-runs) vs new (successful-runs-only) mean cost gap per
(model, interface) cell, before regenerating the table. Confirms the fix
only moves the contaminated text-interface cells and leaves the fc cells
(and closed models) essentially unchanged.
"""
import json
import statistics as st
from collections import defaultdict

rows = [json.loads(l) for l in open("data/runs/exp1.jsonl") if l.strip()]
rows = [r for r in rows if r.get("scenario") == "multi" and r.get("score")]

cells = defaultdict(list)
for r in rows:
    cells[(r["model"], r["interface"])].append(r["score"])

def mean_gap(scores, success_only):
    gaps = [s["cost_gap"] for s in scores
            if s["cost_gap"] is not None
            and (s.get("success") if success_only else True)]
    return round(100 * st.mean(gaps), 3) if gaps else None

print(f"{'model/iface':24}{'OLD gap%':>10}{'NEW gap%':>10}{'n_ok':>7}{'n_all':>7}")
print("-" * 58)
for (m, i), sc in sorted(cells.items()):
    old = mean_gap(sc, success_only=False)
    new = mean_gap(sc, success_only=True)
    n_ok = sum(1 for s in sc if s.get("success") and s["cost_gap"] is not None)
    print(f"{m+'/'+i:24}{str(old):>10}{str(new):>10}{n_ok:>7}{len(sc):>7}")

print("\nInterpretation:")
print("  fc rows: OLD ~= NEW (few/no violations) -> closed models unchanged.")
print("  text rows: negatives disappear; NEW gap = quality of the schedules")
print("  that were actually valid. n_ok shows the denominator behind each.")
