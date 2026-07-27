import json, collections

PATH = "data/runs/exp2.jsonl"
rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]

def fam(scenario):
    s = str(scenario)
    # S1a..S1d -> S1, S2a/S2b -> S2, etc.  keep the family letter
    return s[:2] if len(s) >= 2 else s

errs = [r for r in rows if r.get("error")]
print(f"total rows:     {len(rows)}")
print(f"genuine errors: {len(errs)}")
print()

# --- success by model, keyed off the REAL field ---
print("=== SUCCESS RATE by model (score.success) ===")
by_model = collections.defaultdict(lambda: [0, 0])
for r in rows:
    sc = r.get("score") or {}
    succ = sc.get("success", False)
    by_model[r.get("model")][0] += 1 if succ else 0
    by_model[r.get("model")][1] += 1
for m, (s, n) in sorted(by_model.items()):
    print(f"  {m:12s} {s}/{n}  ({100*s/n:.0f}%)")
print()

# --- failures by model x scenario family (the real finding) ---
print("=== FAILURES (success=False) by model x scenario family ===")
fails = collections.Counter()
totals = collections.Counter()
for r in rows:
    sc = r.get("score") or {}
    f = fam(r.get("scenario"))
    totals[(r.get("model"), f)] += 1
    if not sc.get("success", False):
        fails[(r.get("model"), f)] += 1
for key in sorted(totals):
    if fails[key]:
        print(f"  {key[0]:12s} {key[1]:4s}  {fails[key]}/{totals[key]} failed")
if not fails:
    print("  (no failures anywhere)")
print()

# --- S4 infeasibility handling (inverted: correct = REPORTED, not committed) ---
print("=== S4 infeasibility handling (correct = reported infeasible) ===")
s4 = collections.defaultdict(lambda: {"reported": 0, "fabricated": 0, "n": 0})
for r in rows:
    sc = r.get("score") or {}
    if sc.get("scenario_infeasible"):
        d = s4[r.get("model")]
        d["n"] += 1
        if sc.get("infeasibility_reported"):
            d["reported"] += 1
        if sc.get("fabricated_schedule"):
            d["fabricated"] += 1
for m, d in sorted(s4.items()):
    print(f"  {m:12s} reported {d['reported']}/{d['n']}  fabricated {d['fabricated']}/{d['n']}")
print()

# --- constraint violations detail (S2 power cap, S5 instruction conflict) ---
print("=== rows with constraint_violations logged ===")
viol = collections.Counter()
for r in rows:
    sc = r.get("score") or {}
    if sc.get("constraint_violations"):
        viol[(r.get("model"), fam(r.get("scenario")))] += 1
for key in sorted(viol):
    print(f"  {key[0]:12s} {key[1]:4s}  {viol[key]} runs with violations")
if not viol:
    print("  (none logged)")
