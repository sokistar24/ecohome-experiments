"""
Rebuild data/results/exp2_main.csv from data/runs/exp2.jsonl (all 5 models).

Schema written (one row per model x prompt x family):
    model, prompt, family, success_rate, infeasibility_reported_rate

Conventions (matching the paper / Table 5 / Fig 3):
  - family is the scenario prefix S1..S6 mapped to the long key the plot uses.
  - For S1,S2,S3,S5,S6: success_rate = mean(score.success); infeasibility col blank.
  - For S4: infeasibility_reported_rate = mean(reported AND not fabricated);
    success col blank.
  - prompt values are 'v1' and 'v2-guided' (as in the jsonl prompt_version).
Run from the ecohome_experiments folder:
    python build_exp2_main_csv.py
"""
import json, csv, os, collections

SRC = "data/runs/exp2.jsonl"
OUT = "data/results/exp2_main.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# scenario-prefix -> long family key expected by plot_fig3_taxonomy.py
FAMKEY = {
    "S1": "deadline_conflict",
    "S2": "power_cap",
    "S3": "irregular_calendar",
    "S4": "infeasible",
    "S5": "instruction_vs_calendar",
    "S6": "tool_failure",
}
S4_METRIC = "infeasibility_reported_rate"

def fam_prefix(scn):
    s = str(scn)
    return s[:2] if len(s) >= 2 and s[0] == "S" else s

def correct(r):
    sc = r.get("score") or {}
    if fam_prefix(r.get("scenario")) == "S4":
        return bool(sc.get("infeasibility_reported")) and not bool(sc.get("fabricated_schedule"))
    return bool(sc.get("success"))

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]

# aggregate: (model, prompt, prefix) -> list of correct-bools
agg = collections.defaultdict(list)
models_seen = set()
for r in rows:
    m = r.get("model")
    models_seen.add(m)
    p = str(r.get("prompt_version"))
    pref = fam_prefix(r.get("scenario"))
    if pref not in FAMKEY:
        continue
    agg[(m, p, pref)].append(correct(r))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "prompt", "family", "success_rate", "infeasibility_reported_rate"])
    # stable order
    model_order = ["gpt", "gemini", "claude", "llama-3.3", "qwen-3"]
    models = [m for m in model_order if m in models_seen] + \
             [m for m in sorted(models_seen) if m not in model_order]
    for m in models:
        for pref in ["S1", "S2", "S3", "S4", "S5", "S6"]:
            for p in ["v1", "v2-guided"]:
                vals = agg.get((m, p, pref), [])
                if not vals:
                    continue
                rate = sum(vals) / len(vals)
                if pref == "S4":
                    w.writerow([m, p, FAMKEY[pref], "", f"{rate:.4f}"])
                else:
                    w.writerow([m, p, FAMKEY[pref], f"{rate:.4f}", ""])

print(f"Wrote {OUT}")
print("Models included:", sorted(models_seen))
