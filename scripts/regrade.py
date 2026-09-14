#!/usr/bin/env python3
"""
regrade.py — re-grade saved benchmark outputs with the harness's CURRENT answer-extraction
rule (gsm8k-bench.py: extract_answer), without re-running anything.

  scripts/regrade.py results/swift              # every conc*/ and single/ dir under it
  scripts/regrade.py results/swift/conc8 --dry-run

For each runN.jsonl it writes runN.regrade.jsonl (qid, gold, pred_v1, pred, correct) and updates
runN.summary.json: em_pct/em_correct become the re-graded values, the live values are kept as
em_pct_live/em_correct_live, and answer_extraction records the rule. The original summary is
backed up once to runN.summary.live.json. The raw runN.jsonl is never modified.
It then prints, per run, live vs re-graded EM, and every residual miss (question missed in >= 1
run) with the tail of the model's answer so a human can classify it (model error vs grading).
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import shutil
import sys

import re
RUN_RE = re.compile(r"^run\d+\.jsonl$")   # not runN.regrade.jsonl / runN.metrics-samples.jsonl
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "gsm8k-bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def regrade_dir(d, dry_run, show):
    runs = sorted(p for p in glob.glob(os.path.join(d, "run*.jsonl")) if RUN_RE.match(os.path.basename(p)))
    if not runs:
        return
    print(f"\n== {d}")
    print(f"{'run':<6}{'live EM':>9}{'regraded':>10}{'n':>5}   changed")
    miss = collections.Counter()
    tails = {}
    for path in runs:
        rows = [json.loads(ln) for ln in open(path) if ln.strip()]
        graded = [r for r in rows if r.get("gold") is not None and not r.get("error")]
        out, changed = [], 0
        for r in graded:
            pred = bench.extract_answer(r.get("content") or "")
            ok = (pred == r["gold"])
            if bool(r.get("correct")) != ok:
                changed += 1
            out.append({"qid": r["qid"], "gold": r["gold"], "pred_v1": r.get("pred"), "pred": pred, "correct": ok})
            if not ok:
                miss[r["qid"]] += 1
                tails.setdefault(r["qid"], (r["gold"], pred, (r.get("content") or "")[-140:].replace("\n", " ")))
        live = sum(1 for r in graded if r.get("correct"))
        new = sum(1 for o in out if o["correct"])
        n = len(graded)
        print(f"{os.path.basename(path)[:-6]:<6}{100.0*live/n:>8.1f}%{100.0*new/n:>9.1f}%{n:>5}   {changed}")
        if not dry_run:
            with open(path.replace(".jsonl", ".regrade.jsonl"), "w") as f:
                for o in out:
                    f.write(json.dumps(o) + "\n")
            sp = path.replace(".jsonl", ".summary.json")
            if os.path.exists(sp):
                bak = path.replace(".jsonl", ".summary.live.json")
                if not os.path.exists(bak):
                    shutil.copy(sp, bak)
                with open(sp) as f:
                    smry = json.load(f)
                smry.setdefault("em_pct_live", smry.get("em_pct"))
                smry.setdefault("em_correct_live", smry.get("em_correct"))
                smry["em_correct"], smry["em_n"], smry["em_pct"] = new, n, round(100.0 * new / n, 2)
                smry["answer_extraction"] = bench.EXTRACTION_RULE
                with open(sp, "w") as f:
                    json.dump(smry, f, indent=2)
    if miss:
        print(f"-- residual misses ({len(miss)} questions; runs missed / {len(runs)}) — classify by hand: model error vs grading")
        for q in sorted(miss, key=lambda k: (-miss[k], k))[:show]:
            g, p, t = tails[q]
            print(f"   q{q:<4} {miss[q]}/{len(runs)}  gold={g:<8} pred={p!s:<8} | …{t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="results/<label> or a single run dir")
    ap.add_argument("--dry-run", action="store_true", help="print only; write nothing")
    ap.add_argument("--show", type=int, default=40, help="max residual misses to list per dir")
    a = ap.parse_args()
    dirs = [a.root] if glob.glob(os.path.join(a.root, "run*.jsonl")) else \
        sorted(glob.glob(os.path.join(a.root, "conc*"))) + [os.path.join(a.root, "single")]
    print(f"extraction rule: {bench.EXTRACTION_RULE}")
    for d in dirs:
        if os.path.isdir(d):
            regrade_dir(d, a.dry_run, a.show)
    if not a.dry_run:
        print("\nsummaries updated (live values kept as em_pct_live); re-run `report.py summarize` / `compare` to refresh tables")


if __name__ == "__main__":
    sys.exit(main())
