import json, collections

PATH = "data/runs/exp2.jsonl"

rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]

errs = [r for r in rows if r.get("error")]
ok_false = [r for r in rows if not r.get("error") and not r.get("ok", False)]

print(f"total rows:      {len(rows)}")
print(f"genuine errors:  {len(errs)}")
print(f"ok=False (data): {len(ok_false)}")
print()

# Split genuine errors, flagging context-length ones (the Qwen guided-prompt risk)
if errs:
    ctx = [r for r in errs if "max_model_len" in str(r.get("error", "")).lower()
           or "context" in str(r.get("error", "")).lower()
           or "maximum context" in str(r.get("error", "")).lower()]
    other = [r for r in errs if r not in ctx]
    print("--- GENUINE ERRORS (need re-run) ---")
    if ctx:
        print(f"  context-length errors: {len(ctx)}  <-- Qwen guided-prompt trim needed")
        for r in ctx[:10]:
            print("   ", r.get("model"), r.get("scenario"), r.get("prompt_version"),
                  "->", str(r.get("error"))[:70])
    if other:
        print(f"  other errors (likely 429, just re-run): {len(other)}")
        for r in other[:10]:
            print("   ", r.get("model"), r.get("scenario"), r.get("prompt_version"),
                  "->", str(r.get("error"))[:70])
else:
    print("No genuine errors. Clean run.")
print()

# ok=False breakdown by model x scenario-family, so you can eyeball the safety findings
print("--- ok=False by model x scenario (this is the FINDING, not errors) ---")
by = collections.Counter()
for r in ok_false:
    fam = str(r.get("scenario", "?")).split("/")[0] if "/" in str(r.get("scenario","")) else r.get("scenario","?")
    by[(r.get("model"), fam)] += 1
for (model, fam), n in sorted(by.items()):
    print(f"  {model:12s} {fam:16s} {n}")
