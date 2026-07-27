import json

PATH = "data/runs/exp2.jsonl"
rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]

r = rows[0]
print("=== TOP-LEVEL KEYS in a row ===")
for k in r.keys():
    v = r[k]
    if isinstance(v, (dict, list)):
        print(f"  {k}: <{type(v).__name__}>")
    else:
        print(f"  {k}: {repr(v)[:80]}")

print()
print("=== nested dict contents (if any) ===")
for k, v in r.items():
    if isinstance(v, dict):
        print(f"  {k} -> keys: {list(v.keys())}")

print()
print("=== sample of 3 rows, likely success/score fields ===")
# print whatever looks like a result across a few candidate keys
candidates = ["ok", "success", "passed", "compliant", "gap", "score", "result", "committed", "infeasibility_reported"]
for row in rows[:3]:
    picked = {k: row.get(k) for k in candidates if k in row}
    print(" ", row.get("model"), row.get("scenario"), row.get("prompt_version"), "->", picked)
