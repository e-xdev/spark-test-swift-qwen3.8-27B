#!/usr/bin/env python3
"""
report.py — turn gsm8k-bench.py runs into the tables that go in REPORT.md.

  report.py summarize results/swift                 # median / min / max across runs, per mode dir
  report.py compare results/swift results/base \
      --labels "Swift-NVFP4 (UkisAI)" "Qwen3.8-27B-NVFP4 (unsloth)" --with-card

Layout expected:  results/<label>/conc8/run*.summary.json   (8-concurrent GSM8K, card protocol; any conc*/ works)
                  results/<label>/single/run*.summary.json  (single-stream row, clearly labeled)
                  results/<label>/smoke/                    (garbage/coherence smoke test, not scored here)

Every cell is "median (min–max)" over the measured runs unless stated otherwise.
Stdlib only.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

RUN_RE = re.compile(r"^run\d+\.jsonl$")   # raw per-question outputs only (not regrade / metrics-samples files)

# (key in run summary, row label, format)
ROWS = [
    ("em_pct",                              "GSM8K exact match, %",                               "{:.1f}"),
    ("truncated",                           "Truncated at max_tokens (count)",                    "{:.0f}"),
    ("mean_completion_tokens",              "Mean completion tokens (reasoning + answer)",        "{:.0f}"),
    ("median_decode_tps",                   "Median tok/s per request, decode phase",             "{:.1f}"),
    ("server_gen_tps_steady_median",        "Engine aggregate generation tok/s, steady state (/metrics, running >= concurrency)", "{:.1f}"),
    ("server_gen_tps_run_aggregate",        "Engine aggregate generation tok/s over the whole run (incl. ramp/tail)", "{:.1f}"),
    ("median_latency_s",                    "Median time-to-final-answer per question, s",        "{:.1f}"),
    ("p90_latency_s",                       "p90 time-to-final-answer per question, s",           "{:.1f}"),
    ("wall_s",                              "Wall time for the whole batch, s (all questions of the run)", "{:.0f}"),
    ("median_ttft_s",                       "Median TTFT, s",                                     "{:.2f}"),
    ("mtp_accept_rate_token_weighted_pct",  "MTP draft-token acceptance, % (engine counters, token-weighted)", "{:.1f}"),
    ("mtp_mean_acceptance_length_median",   "MTP mean acceptance length",                         "{:.2f}"),
    ("n_errors",                            "Request errors (count)",                             "{:.0f}"),
]

# Values transcribed from https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4 on 2026-09-13
# (RTX PRO 6000 Blackwell, SM120, native FP4 path, vLLM 0.29.0). RE-VERIFY against the live card before publishing.
CARD_REFERENCE = {
    "em_pct": "96.5",
    "truncated": "0",
    "mean_completion_tokens": "391",
    "median_decode_tps": "85 (card: 'median tokens/s per request, 8 concurrent')",
    "mtp_accept_rate_token_weighted_pct": "61",
}
CARD_LABEL = "Swift-NVFP4, UkisAI card (RTX PRO 6000 Blackwell SM120, vLLM 0.29.0)"


def load_runs(d):
    runs = []
    for p in sorted(glob.glob(os.path.join(d, "run*.summary.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def agg(runs, key):
    xs = [r[key] for r in runs if r.get(key) is not None]
    if not xs:
        return None
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def fmt_cell(a, fmt):
    if not a:
        return "n/a"
    m = fmt.format(a["median"])
    if a["n"] == 1:
        return m
    return f"{m} ({fmt.format(a['min'])}–{fmt.format(a['max'])})"


def summarize_dir(d):
    runs = load_runs(d)
    if not runs:
        return None
    out = {"dir": d, "n_runs": len(runs), "model": runs[0].get("model"), "label": runs[0].get("label"),
           "concurrency": runs[0].get("concurrency"), "n_per_run": runs[0].get("n_requested"),
           "effort": runs[0].get("effort"), "max_tokens": runs[0].get("max_tokens"), "metrics": {}}
    for key, _, _ in ROWS:
        out["metrics"][key] = agg(runs, key)
    pp = [r["mtp_per_position_mean"] for r in runs if r.get("mtp_per_position_mean")]
    if pp:
        w = min(len(p) for p in pp)
        out["mtp_per_position_mean"] = [round(statistics.mean(p[i] for p in pp), 3) for i in range(w)]
    return out


def md_summary(s):
    lines = [f"### {s['label']} — `{s['model']}` — {os.path.basename(s['dir'])}: "
             f"{s['n_runs']} runs × {s['n_per_run']} requests @ concurrency {s['concurrency']}, "
             f"max_tokens {s['max_tokens']}, effort {s['effort']}", "",
             "| Metric | median (min–max) |", "|---|---|"]
    for key, label, fmt in ROWS:
        lines.append(f"| {label} | {fmt_cell(s['metrics'].get(key), fmt)} |")
    if s.get("mtp_per_position_mean"):
        lines.append(f"| MTP per-position acceptance (mean) | {', '.join(str(x) for x in s['mtp_per_position_mean'])} |")
    return "\n".join(lines) + "\n"


def cmd_summarize(args):
    root = args.dir
    modes = sorted(glob.glob(os.path.join(root, "conc*"))) + [os.path.join(root, "single")]
    modes = [m for m in modes if os.path.isdir(m)]
    if not modes and glob.glob(os.path.join(root, "run*.summary.json")):
        modes = [root]
    if not modes:
        sys.exit(f"no run*.summary.json under {root} (expected conc8/ and/or single/)")
    md, js = [], {}
    for d in modes:
        s = summarize_dir(d)
        if s:
            md.append(md_summary(s))
            js[os.path.basename(d)] = s
    text = "\n".join(md)
    print(text)
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write(text)
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(js, f, indent=2)
    print(f"-> {root}/summary.md, summary.json", file=sys.stderr)


def paired_outcomes(dir_a, dir_b):
    """Pair run k of A with run k of B on the same qids; count both/only-A/only-B/neither
    over all paired question instances. Also per-question 'correct in how many runs'."""
    ra = sorted(p for p in glob.glob(os.path.join(dir_a, "run*.jsonl")) if RUN_RE.match(os.path.basename(p)))
    rb = sorted(p for p in glob.glob(os.path.join(dir_b, "run*.jsonl")) if RUN_RE.match(os.path.basename(p)))
    k = min(len(ra), len(rb))
    tot = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0, "paired_instances": 0, "runs_paired": k}
    per_q = {}
    for i in range(k):
        ca, cb = {}, {}
        for path, dst in ((ra[i], ca), (rb[i], cb)):
            rg = path.replace(".jsonl", ".regrade.jsonl")   # written by regrade.py; same extraction rule for both sides
            with open(rg if os.path.exists(rg) else path) as f:
                for ln in f:
                    r = json.loads(ln)
                    if "correct" in r:
                        dst[r["qid"]] = bool(r["correct"])
        for q in sorted(set(ca) & set(cb)):
            a, b = ca[q], cb[q]
            tot["paired_instances"] += 1
            tot["both" if (a and b) else "only_a" if a else "only_b" if b else "neither"] += 1
            pq = per_q.setdefault(q, [0, 0, 0])
            pq[0] += a
            pq[1] += b
            pq[2] += 1
    tot["questions_a_never_right"] = sorted(q for q, v in per_q.items() if v[0] == 0)
    tot["questions_b_never_right"] = sorted(q for q, v in per_q.items() if v[1] == 0)
    return tot


def conc_dir(root):
    """The concurrent-run dir (conc8 by default; whatever conc* exists)."""
    ds = sorted(glob.glob(os.path.join(root, "conc*")))
    if not ds:
        sys.exit("no conc*/ dir under %s" % root)
    if len(ds) > 1:
        print("[warn] several conc* dirs under %s, using %s" % (root, ds[-1]), file=sys.stderr)
    return ds[-1]


def cmd_compare(args):
    la, lb = args.labels
    ca = conc_dir(args.a)
    cb = conc_dir(args.b)
    sa, sb = summarize_dir(ca), summarize_dir(cb)
    if not sa or not sb:
        sys.exit("both sides need conc*/run*.summary.json")
    if sa["concurrency"] != sb["concurrency"] or sa["n_per_run"] != sb["n_per_run"]:
        print("[warn] protocol mismatch: A conc=%s n=%s vs B conc=%s n=%s"
              % (sa["concurrency"], sa["n_per_run"], sb["concurrency"], sb["n_per_run"]), file=sys.stderr)
    ssa, ssb = summarize_dir(os.path.join(args.a, "single")), summarize_dir(os.path.join(args.b, "single"))
    po = paired_outcomes(ca, cb)

    rule = "see run summaries"
    for d in (ca, cb):
        for r in load_runs(d):
            if r.get("answer_extraction"):
                rule = r["answer_extraction"]
    hdr = ["Metric", la, lb] + ([CARD_LABEL] if args.with_card else [])
    lines = [f"## {la} vs {lb} — same DGX Spark (GB10, SM121, native FlashInfer CUTLASS NVFP4 kernel), same vLLM, same protocol", "",
             f"Protocol: GSM8K test first {sa['n_per_run']} questions, exact match; {sa['n_runs']} measured runs "
             f"(median, min–max); concurrency {sa['concurrency']}; max_tokens {sa['max_tokens']}; "
             f"effort {sa['effort']}; sampling temp 1.0 / top_p 0.95 / top_k 20 / min_p 0; MTP method=mtp, 3 draft tokens.",
             "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for key, label, fmt in ROWS:
        row = [label, fmt_cell(sa["metrics"].get(key), fmt), fmt_cell(sb["metrics"].get(key), fmt)]
        if args.with_card:
            row.append(CARD_REFERENCE.get(key, "—"))
        lines.append("| " + " | ".join(row) + " |")
    pair = f"{po['both']} / {po['only_a']} / {po['only_b']} (neither: {po['neither']}; {po['paired_instances']} paired instances over {po['runs_paired']} runs)"
    paired_row = f"| Paired outcome (both right / only {la} / only {lb}) | {pair} | ← same |"
    if args.with_card:
        paired_row += " 192 / 4 / 1 (card: BF16 vs NVFP4) |"
    lines.append(paired_row)
    if sa.get("mtp_per_position_mean") and sb.get("mtp_per_position_mean"):
        lines.append(f"| MTP per-position acceptance (mean) | {sa['mtp_per_position_mean']} | {sb['mtp_per_position_mean']} |" + (" — |" if args.with_card else ""))
    if ssa and ssb:
        lines.append(f"| **Single-stream** median decode tok/s (concurrency 1, {ssa['n_per_run']} q × {ssa['n_runs']} runs) | "
                     f"{fmt_cell(ssa['metrics'].get('median_decode_tps'), '{:.1f}')} | {fmt_cell(ssb['metrics'].get('median_decode_tps'), '{:.1f}')} |" + (" not on card |" if args.with_card else ""))
        lines.append(f"| **Single-stream** MTP acceptance, % | "
                     f"{fmt_cell(ssa['metrics'].get('mtp_accept_rate_token_weighted_pct'), '{:.1f}')} | {fmt_cell(ssb['metrics'].get('mtp_accept_rate_token_weighted_pct'), '{:.1f}')} |" + (" — |" if args.with_card else ""))
    lines.append("| Weights in GPU memory (startup log `Model loading took N GiB`) | see results/<label>/startup-lines.txt | see results/<label>/startup-lines.txt |" + (" ~29 GB |" if args.with_card else ""))
    lines += ["",
              "Notes: per-request tok/s is decode-phase (first→last streamed token), so TTFT is excluded; the engine's own "
              "generation-token counters (/metrics, sampled every 10 s) give the aggregate over all running requests and are "
              "reported separately. MTP figures are token-weighted deltas of the engine's spec-decode counters over each run. "
              f"Exact-match extraction rule: {rule}. Raw per-question outputs and engine-counter snapshots are in each results dir.",
              ""]
    if po["questions_a_never_right"] or po["questions_b_never_right"]:
        lines.append(f"Questions never solved in any run — {la}: {po['questions_a_never_right']}; {lb}: {po['questions_b_never_right']}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    out = args.out or os.path.join(os.path.dirname(args.a.rstrip("/")) or ".", "compare.md")
    with open(out, "w") as f:
        f.write(text)
    with open(out.replace(".md", ".json"), "w") as f:
        json.dump({"a": sa, "b": sb, "single_a": ssa, "single_b": ssb, "paired": po}, f, indent=2)
    print(f"-> {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_summarize)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--labels", nargs=2, default=["Swift-NVFP4 (UkisAI)", "Qwen3.8-27B-NVFP4 (unsloth)"])
    c.add_argument("--with-card", action="store_true", help="add the UkisAI card's SM120 column (transcribed values)")
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
