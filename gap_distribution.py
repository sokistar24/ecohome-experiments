"""Distribution of cost gaps for the open models' fc runs (multi scenario),
to distinguish 'a few large outliers inflate the mean' from 'systematically
worse windows'. Prints median/mean/max and the worst individual runs.
"""
import json
import statistics as st
from collections import defaultdict

rows = [json.loads(l) for l in open("data/runs/exp1.jsonl") if l.strip()]
rows = [r for r in rows if r.get("scenario") == "multi" and r.get("score")]

def gaps_for(model, iface, success_only=True):
    out = []
    for r in rows:
        if r["model"] != model or r["interface"] != iface:
            continue
        s = r["score"]
        if s.get("cost_gap") is None:
            continue
        if success_only and not s.get("success"):
            continue
        out.append((s["cost_gap"], r["day"]))
    return out

for model in ["llama-3.3", "qwen-3", "claude"]:
    for iface in ["fc", "text"]:
        g = gaps_for(model, iface)
        if not g:
            print(f"{model}/{iface}: no successful scored runs"); continue
        vals = sorted(x[0] for x in g)
        pct = [round(100*v, 2) for v in vals]
        median = round(100*st.median(vals), 3)
        mean = round(100*st.mean(vals), 3)
        mx = round(100*max(vals), 2)
        n_big = sum(1 for v in vals if v > 0.05)   # >5% gap
        print(f"{model}/{iface:4} n={len(vals):2}  median={median:>6}  "
              f"mean={mean:>6}  max={mx:>6}  runs>5%={n_big}")
    # worst fc runs for the open models
    if model in ("llama-3.3", "qwen-3"):
        worst = sorted(gaps_for(model, "fc"), reverse=True)[:5]
        print(f"    worst {model} fc runs: " +
              ", ".join(f"{round(100*v,1)}% ({d})" for v, d in worst))
    print()

print("Read: if median is small but mean/max are large with a few runs>5%,")
print("the open-model fc gap is driven by a handful of bad-window outliers,")
print("not a systematic deficit. Report median alongside mean, or note the")
print("outliers explicitly.")
