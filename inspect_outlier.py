"""Inspect the Qwen fc 122.86% outlier (2026-06-07) and any other gaps>10%,
to determine whether they are ratio-instability on low/negative-cost days
(|J*| tiny) rather than genuinely catastrophic schedules.
"""
import json

rows = [json.loads(l) for l in open("data/runs/exp1.jsonl") if l.strip()]
rows = [r for r in rows if r.get("scenario") == "multi" and r.get("score")]

big = []
for r in rows:
    s = r["score"]
    g = s.get("cost_gap")
    if g is not None and abs(g) > 0.10:      # >10%
        big.append(r)

print(f"runs with |gap|>10%: {len(big)}\n")
for r in sorted(big, key=lambda r: abs(r["score"]["cost_gap"]), reverse=True):
    s = r["score"]
    print(f"{r['model']}/{r['interface']}/{r['day']}  gap={round(100*s['cost_gap'],1)}%")
    print(f"    agent_cost={s.get('agent_cost')}  milp_cost={s.get('milp_cost')}"
          f"  success={s.get('success')}")
    print(f"    committed={s.get('committed_starts')}")
    print(f"    milp     ={s.get('milp_starts')}")
    ac, mc = s.get('agent_cost'), s.get('milp_cost')
    if mc is not None and abs(mc) < 0.5:
        print(f"    >> |milp_cost|={abs(mc):.4f} is small -> gap RATIO is unstable")
        print(f"       here (small denominator inflates %). Absolute miss ="
              f" {round((ac-mc),4) if ac is not None else '?'} GBP.")
    print()
