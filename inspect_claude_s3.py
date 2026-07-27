import json

PATH = "data/runs/exp2.jsonl"
rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]

# Claude S3 runs (S3a, S3b, S3c) — both prompt versions, all reps
claude_s3 = [
    r for r in rows
    if r.get("model") == "claude"
    and str(r.get("scenario", "")).startswith("S3")
]

# sort for stable reading: scenario, prompt, then whatever rep marker exists
claude_s3.sort(key=lambda r: (str(r.get("scenario")), str(r.get("prompt_version")), str(r.get("date", "")), str(r.get("rep", ""))))

print(f"Found {len(claude_s3)} Claude S3 runs\n")
print("=" * 90)

for r in claude_s3:
    sc = r.get("score") or {}
    scenario = r.get("scenario")
    prompt = r.get("prompt_version")
    date = r.get("date", "?")
    rep = r.get("rep", "?")
    success = sc.get("success")

    print(f"\n### {scenario} | prompt={prompt} | date={date} | rep={rep} | success={success}")
    print(f"    committed_starts : {sc.get('committed_starts')}")
    print(f"    milp_starts      : {sc.get('milp_starts')}")
    print(f"    deadline_satisfied: {sc.get('deadline_satisfied')}")
    print(f"    constraint_violations: {sc.get('constraint_violations')}")
    print(f"    cost_gap         : {sc.get('cost_gap')}  optimal={sc.get('optimal')}")
    print(f"    milp_cost/agent  : {sc.get('milp_cost')} / {sc.get('agent_cost')}")

    final = r.get("final") or ""
    # show the agent's own explanation of which deadline/event it used
    print(f"    --- agent final message ---")
    print("    " + str(final)[:900].replace("\n", "\n    "))
    print("-" * 90)

# Compact summary: where do committed and MILP starts actually diverge?
print("\n\n=== DIVERGENCE SUMMARY (committed vs MILP per appliance) ===")
for r in claude_s3:
    sc = r.get("score") or {}
    cs = sc.get("committed_starts") or {}
    ms = sc.get("milp_starts") or {}
    diffs = {k: (cs.get(k), ms.get(k)) for k in ms if cs.get(k) != ms.get(k)}
    tag = f"{r.get('scenario')}/{r.get('prompt_version')}/rep{r.get('rep','?')}"
    if diffs:
        print(f"  {tag:24s} DIFFERS: {diffs}   success={sc.get('success')}")
    else:
        print(f"  {tag:24s} matches MILP exactly   success={sc.get('success')}")
