import json

PATH = "data/runs/exp2.jsonl"
rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]

MODELS = ["gpt", "llama-3.3"]

def s3_rows(model):
    return [r for r in rows
            if r.get("model") == model
            and str(r.get("scenario", "")).startswith("S3")]

for model in MODELS:
    runs = s3_rows(model)
    runs.sort(key=lambda r: (str(r.get("scenario")), str(r.get("prompt_version"))))
    print("=" * 84)
    print(f"MODEL: {model}   ({len(runs)} S3 runs)")
    print("=" * 84)
    for r in runs:
        sc = r.get("score") or {}
        cs = sc.get("committed_starts") or {}
        ms = sc.get("milp_starts") or {}
        ev_c, ev_m = cs.get("ev_charger"), ms.get("ev_charger")
        succ = sc.get("success")
        viol = sc.get("constraint_violations") or []
        gap = sc.get("cost_gap")

        cls = ""
        if ev_c is not None and ev_m is not None and not succ:
            delta = ev_c - ev_m
            has_deadline = any("deadline" in str(v).lower() for v in viol)
            if delta == 1 and has_deadline:
                cls = "  [PRECISION off-by-one]"
            elif delta >= 2 and has_deadline:
                cls = f"  [GROSS +{delta} slots = {delta*0.5:.1f}h late]"
            elif has_deadline:
                cls = f"  [deadline viol, delta={delta}]"

        tag = f"{r.get('scenario')}/{r.get('prompt_version')}"
        print(f"  {tag:18s} success={str(succ):5s} EV committed={ev_c} MILP={ev_m} "
              f"gap={gap} viol={viol}{cls}")
    print()

print("=" * 84)
print("VERDICT: failure class on S3b, and guided-prompt repair")
print("=" * 84)
for model in MODELS:
    runs = s3_rows(model)
    precision = gross = 0
    v1_fail = v2_fail = 0
    for r in runs:
        sc = r.get("score") or {}
        scenario = str(r.get("scenario"))
        prompt = str(r.get("prompt_version"))
        if not scenario.startswith("S3b"):
            continue
        cs = sc.get("committed_starts") or {}
        ms = sc.get("milp_starts") or {}
        ev_c, ev_m = cs.get("ev_charger"), ms.get("ev_charger")
        succ = sc.get("success")
        if not succ:
            if "v1" in prompt:
                v1_fail += 1
            else:
                v2_fail += 1
            if ev_c is not None and ev_m is not None:
                delta = ev_c - ev_m
                if delta == 1:
                    precision += 1
                elif delta >= 2:
                    gross += 1
    print(f"  {model:10s}: precision off-by-one={precision}  gross={gross}  "
          f"| S3b fails: v1={v1_fail}/3  v2-guided={v2_fail}/3")
