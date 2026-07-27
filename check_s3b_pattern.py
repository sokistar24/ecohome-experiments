import json

PATH = "data/runs/exp2.jsonl"
rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]

MODELS = ["gemini", "qwen-3"]

def s3_rows(model):
    return [r for r in rows
            if r.get("model") == model
            and str(r.get("scenario", "")).startswith("S3")]

for model in MODELS:
    runs = s3_rows(model)
    runs.sort(key=lambda r: (str(r.get("scenario")), str(r.get("prompt_version"))))
    print("=" * 80)
    print(f"MODEL: {model}   ({len(runs)} S3 runs)")
    print("=" * 80)

    for r in runs:
        sc = r.get("score") or {}
        scenario = r.get("scenario")
        prompt = r.get("prompt_version")
        cs = sc.get("committed_starts") or {}
        ms = sc.get("milp_starts") or {}
        ev_c = cs.get("ev_charger")
        ev_m = ms.get("ev_charger")
        succ = sc.get("success")
        viol = sc.get("constraint_violations") or []
        gap = sc.get("cost_gap")

        # Is this the zero-slack off-by-one signature? EV one slot late + deadline violation
        offbyone = (ev_c is not None and ev_m is not None
                    and ev_c == ev_m + 1
                    and any("deadline" in str(v).lower() for v in viol))

        tag = f"{scenario}/{prompt}"
        flag = "  <-- ZERO-SLACK OFF-BY-ONE" if offbyone else ""
        print(f"  {tag:18s} success={str(succ):5s} EV committed={ev_c} MILP={ev_m} "
              f"gap={gap} viol={viol}{flag}")
    print()

# Compact verdict
print("=" * 80)
print("VERDICT: does each model's S3 failure = the S3b zero-slack off-by-one?")
print("=" * 80)
for model in MODELS:
    runs = s3_rows(model)
    s3b_fails_offbyone = 0
    s3b_total = 0
    non_s3b_fails = 0
    for r in runs:
        sc = r.get("score") or {}
        scenario = str(r.get("scenario"))
        cs = sc.get("committed_starts") or {}
        ms = sc.get("milp_starts") or {}
        ev_c, ev_m = cs.get("ev_charger"), ms.get("ev_charger")
        viol = sc.get("constraint_violations") or []
        succ = sc.get("success")
        if scenario.startswith("S3b"):
            s3b_total += 1
            if (ev_c is not None and ev_m is not None and ev_c == ev_m + 1
                    and any("deadline" in str(v).lower() for v in viol)):
                s3b_fails_offbyone += 1
        else:
            if not succ:
                non_s3b_fails += 1
    print(f"  {model:10s}: S3b off-by-one failures {s3b_fails_offbyone}/{s3b_total}; "
          f"non-S3b S3 failures: {non_s3b_fails}")
