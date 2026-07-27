"""Diagnose negative cost-gap rows in exp1.jsonl.

A negative gap ("agent cheaper than the MILP optimum") is impossible if
scoring is correct, so it almost always means the committed schedule
violates a hard constraint (e.g. EV finishing after its deadline) and the
gap absorbed that. This prints each negative-gap run with its violations,
committed vs MILP starts, and success flag, then summarises by (model,
interface) and cross-tabs against whether success/violations line up.
"""
import json
from collections import Counter

rows = [json.loads(l) for l in open("data/runs/exp1.jsonl")]
neg = [r for r in rows
       if r.get("score")
       and r["score"].get("cost_gap") is not None
       and r["score"]["cost_gap"] < 0]

print(f"negative-gap rows: {len(neg)}\n")

for r in neg[:20]:
    s = r["score"]
    print(f"  {r['model']}/{r['interface']}/{r['scenario']}/{r['day']} "
          f"gap={s['cost_gap']} success={s.get('success')} "
          f"viol={s.get('constraint_violations')}")
    print(f"      committed={s.get('committed_starts')}")
    print(f"      milp     ={s.get('milp_starts')}")

print("\n--- by (model, interface) ---")
by_cell = Counter((r["model"], r["interface"]) for r in neg)
for k, v in sorted(by_cell.items()):
    print(f"  {k}: {v} negative-gap rows")

print("\n--- do negatives coincide with violations / failure? ---")
with_viol = sum(1 for r in neg if r["score"].get("constraint_violations"))
success_true = sum(1 for r in neg if r["score"].get("success") is True)
print(f"  negative-gap rows WITH constraint_violations: {with_viol}/{len(neg)}")
print(f"  negative-gap rows marked success=True:        {success_true}/{len(neg)}")
if success_true == 0 and with_viol == len(neg):
    print("\n  => VERDICT: scorer is CORRECT. Every negative-gap run violated a")
    print("     constraint and was marked failed. The negative MEAN gap is a")
    print("     reporting artifact: gaps are being averaged over violating runs.")
    print("     Fix is presentational (mean gap over successful runs only).")
else:
    print("\n  => Some negative-gap rows are success=True or have no violation;")
    print("     that points to a genuine scoring bug to investigate, not just")
    print("     a reporting choice.")
