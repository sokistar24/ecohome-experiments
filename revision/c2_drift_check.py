"""
Phase C, step 2 -- provider drift between the direct runs (July) and the hybrid
runs (October).

The direct agent with the guided prompt was rerun in October for repeat 0 of
every Exp 2 scenario and model (grid exp2-drift in revision/hybrid.py). This
script compares those outcomes with the July repeat-0 outcomes (A1 deadline
rule) and, as the reference for ordinary run-to-run variation, with how often
July's repeat 0 agrees with July's repeats 1 and 2.

No LLM calls. Run from the repo root (after the exp2-drift runs):
    python -m revision.c2_drift_check
Output: revision/outputs/c2_drift.txt and c2_drift.csv
"""
from __future__ import annotations

import collections
import copy
import csv
import json

from experiments import config
from experiments.runner import run_id
from revision.a1_rescore_departure_buffer import (
    load_scenarios, detect_buffered, rescore, correct)

MODELS = ["gpt", "gemini", "claude", "llama-3.3", "qwen-3"]
OUT = config.ROOT / "revision" / "outputs"


def main():
    scen = load_scenarios()
    buf = detect_buffered(scen)

    def cons(sid):
        c = copy.deepcopy(scen[sid]["constraints"])
        if sid in buf:
            c["ev_latest_finish"] = buf[sid]
        return c

    july = {}
    for l in open(config.RUNS / "exp2.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["prompt_version"] != "v2-guided":
            continue
        for rep in range(3):
            if run_id(exp="exp2", scenario=r["scenario"], model=r["model"], interface="fc",
                      day=r["day"], rep=rep, pv="v2-guided") == r["run_id"]:
                july[(r["model"], r["scenario"], rep)] = correct(
                    r["scenario"], rescore(r, scen, cons(r["scenario"])))
    oct_ = {}
    dates = set()
    for l in open(config.RUNS / "exp2-drift.jsonl", encoding="utf-8"):
        r = json.loads(l)
        oct_[(r["model"], r["scenario"])] = correct(r["scenario"], rescore(r, scen, cons(r["scenario"])))
        dates.add(r["ts"][:10])

    lines, rows = [], []
    say = lambda s="": (print(s), lines.append(s))
    say(f"C2 -- provider drift check: {len(oct_)} October reruns ({min(dates)} to {max(dates)}) "
        f"vs July repeat 0, direct agent, guided prompt")
    say(f"  {'model':10s} {'July correct':>13s} {'Oct correct':>12s} {'Oct = July':>11s} "
        f"{'July rep0 = rep1/2':>19s}")
    tot = collections.Counter()
    for m in MODELS:
        ks = [k for k in oct_ if k[0] == m]
        jc = sum(july[(m, s, 0)] for _, s in ks)
        oc = sum(oct_[k] for k in ks)
        same = sum(oct_[(m, s)] == july[(m, s, 0)] for _, s in ks)
        within = sum(july[(m, s, 0)] == july[(m, s, r2)] for _, s in ks for r2 in (1, 2))
        tot.update({"n": len(ks), "same": same, "within": within, "within_n": 2 * len(ks),
                    "jc": jc, "oc": oc})
        say(f"  {m:10s} {jc:>9d}/{len(ks):<3d} {oc:>8d}/{len(ks):<3d} {same:>7d}/{len(ks):<3d} "
            f"{within:>14d}/{2 * len(ks):<3d}")
        rows.append({"model": m, "july_correct": jc, "oct_correct": oc, "n": len(ks),
                     "oct_equals_july": same, "july_rep0_equals_rep12": within,
                     "july_pairs": 2 * len(ks)})
    say(f"  {'all':10s} {tot['jc']:>9d}/{tot['n']:<3d} {tot['oc']:>8d}/{tot['n']:<3d} "
        f"{tot['same']:>7d}/{tot['n']:<3d} {tot['within']:>14d}/{tot['within_n']:<3d}")
    say(f"October agrees with July in {100 * tot['same'] / tot['n']:.0f}% of runs; July repeats "
        f"agree with each other in {100 * tot['within'] / tot['within_n']:.0f}%.")
    flips = [(m, s, july[(m, s, 0)], v) for (m, s), v in sorted(oct_.items()) if v != july[(m, s, 0)]]
    say("Changed outcomes (model, scenario, July -> October): "
        + ", ".join(f"{m} {s} {'ok' if a else 'fail'}->{'ok' if b else 'fail'}" for m, s, a, b in flips))
    (OUT / "c2_drift.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT / "c2_drift.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote revision/outputs/c2_drift.txt and c2_drift.csv")


if __name__ == "__main__":
    main()
