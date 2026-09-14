#!/usr/bin/env python3
"""
gsm8k-bench.py — throughput + exact-match harness for a vLLM OpenAI-compatible
server, mirroring the protocol on the ukisai/Swift-Qwen3.8-27B-NVFP4 model card:

    GSM8K test, first 200 questions, exact match
    temperature 1.0 / top_p 0.95 / top_k 20 / min_p 0
    8 concurrent requests, 8192-token output cap
    template-default reasoning effort (no reasoning_effort kwarg unless --effort)
    MTP: method=mtp, num_speculative_tokens=3  (set at `vllm serve` time, not here)

Measurement rules (so the numbers are defensible):
  * ENGINE-SIDE numbers come from vLLM's /metrics endpoint, sampled every 10 s during
    each run (generation tokens, drafts, draft tokens, accepted tokens, running requests).
    Raw snapshots are saved per run. Steady state = windows with running >= concurrency.
  * FALLBACK engine-side speed: "Avg generation throughput" lines captured from the
    tmux pane where vLLM logs (NOT `docker logs`, which is empty under the eugr
    launcher). Only lines with Running >= concurrency count as steady state.
  * ENGINE-SIDE MTP: "SpecDecoding metrics" lines from the same pane, token-weighted.
  * CLIENT-SIDE per-request tok/s is measured over the DECODE phase only
    (first streamed token -> last streamed token), so prefill/TTFT and the HTTP
    round-trip do not leak into it. This is the analogue of the card's
    "Median tokens/s per request (8 concurrent)".
  * Every request carries the card's sampling; the first request body of every
    run is saved verbatim (request-sample.json) as proof of what was sent.
  * Warmup requests (outside the measured question set) precede every measured
    run; N measured runs -> median / min / max in report.py.
  * Raw engine log lines for each run are saved next to the results (provenance).

Stdlib only. Python >= 3.8. Run `--selftest` to check the parsers without a GPU.
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib import request as urlreq

# UkisAI model card sampling — verbatim. Do not change without relabeling the results.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}

# --------------------------------------------------------------------------- answers
# Extraction rule v3 (document it next to any EM number). Applied to the FINAL (non-reasoning) content:
#   0. clean LaTeX: thousands separators 70{,}000 / 7\,200 -> 70000 / 7200; \$ -> $; \( \) \[ \] -> space; \text{...} -> space
#   1. '#### N' (last occurrence)
#   2. last \boxed{...}: first number inside
#   3. last 'final answer' / 'answer is' / 'answer:' on a line: first number after it
#   4. last line containing a bold **...** segment with a number: the last such segment that contains
#      '=' if any, else the first (segments ending in ':' such as '**Step 3:**' are skipped)
#   5. last number in the content
#   Inside the chosen segment (2-4): number after the last '=' > last $-amount > first number.
#   A '-' is only read as a minus sign when it does not follow a word character, so '80-150-170'
#   yields 80, 150, 170. Commas and $ are stripped; numeric compare (36.0 == 36).
# v1 (used live during the first Swift pass) was: #### > last number in last \boxed{} > last number.
# The v1 -> v2 change was motivated by an audit of the misses (see README); regrade.py re-grades saved outputs.
NUM_RE = re.compile(r"(?<!\w)-?\d[\d,]*(?:\.\d+)?")
HASH_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*(?:\*\*)?\s*([^\n]{0,80})", re.IGNORECASE)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
LATEX_SEP_RE = re.compile(r"(?<=\d)(?:\{,\}|\\,|\\!|\\ )(?=\d)")
TEXT_RE = re.compile(r"\\text\{[^{}]*\}")


def norm_num(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return s
    return str(int(v)) if v.is_integer() else ("%g" % v)


def _clean(text):
    t = LATEX_SEP_RE.sub("", text)
    t = t.replace("\\$", "$")
    t = TEXT_RE.sub(" ", t)
    for tok in ("\\(", "\\)", "\\[", "\\]"):
        t = t.replace(tok, " ")
    return t


MONEY_RE = re.compile(r"\$\s*(-?\d[\d,]*(?:\.\d+)?)")


def _pick(seg):
    """The answer inside one segment: number after the last '=' > last $-amount > first number."""
    nums = NUM_RE.findall(seg)
    if not nums:
        return None
    if "=" in seg:
        tail = NUM_RE.findall(seg.rsplit("=", 1)[1])
        if tail:
            return norm_num(tail[0])
    money = MONEY_RE.findall(seg)
    if money:
        return norm_num(money[-1])
    return norm_num(nums[0])


def extract_answer(text):
    if not text:
        return None
    t = _clean(text)
    m = HASH_RE.findall(t)
    if m:
        return norm_num(m[-1])
    m = BOXED_RE.findall(t)
    if m:
        v = _pick(m[-1])
        if v is not None:
            return v
    m = ANSWER_RE.findall(t)
    if m:
        v = _pick(m[-1])
        if v is not None:
            return v
    for line in reversed(t.splitlines()):
        segs = [g for g in BOLD_RE.findall(line) if not g.rstrip().endswith(":") and NUM_RE.search(g)]
        if not segs:
            continue
        eq = [g for g in segs if "=" in g]
        return _pick(eq[-1] if eq else segs[0])
    nums = NUM_RE.findall(t)
    return norm_num(nums[-1]) if nums else None


def extract_answer_v1(text):
    """The rule used live during the first Swift pass, kept for reference/regrade comparison."""
    if not text:
        return None
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return norm_num(m[-1])
    m = BOXED_RE.findall(text)
    if m:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", m[-1])
        if nums:
            return norm_num(nums[-1])
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return norm_num(nums[-1]) if nums else None


def gold_answer(ans):
    return norm_num(ans.split("####")[-1])


EXTRACTION_RULE = "v3: #### > last \\boxed{} > 'answer is/:' > bold on last bolded line (segment with '=' preferred) > last number; inside a segment: number after last '=' > last $-amount > first number; LaTeX {,} separators removed; '-' after a word char is not a minus"


# --------------------------------------------------------------------------- engine log
GEN_RE = re.compile(r"Avg generation throughput:\s*([\d.]+)\s*tokens/s")
PROMPT_RE = re.compile(r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s")
RUN_RE = re.compile(r"Running:\s*(\d+)\s*reqs")
SPEC_MAL_RE = re.compile(r"Mean acceptance length:\s*([\d.]+)")
SPEC_ACC_RE = re.compile(r"Accepted:\s*(\d+)\s*tokens")
SPEC_DRF_RE = re.compile(r"Drafted:\s*(\d+)\s*tokens")
SPEC_POS_RE = re.compile(r"Per-position acceptance rate:\s*([\d.]+(?:\s*,\s*[\d.]+)*)")
SPEC_RATE_RE = re.compile(r"Avg Draft acceptance rate:\s*([\d.]+)%")


def capture_log(cmd):
    """Run the log-capture shell command (default: tmux capture-pane) and return lines."""
    if not cmd:
        return None
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            sys.stderr.write(f"[warn] log capture returned {p.returncode}: {p.stderr.strip()[:200]}\n")
            return None
        lines = p.stdout.splitlines()
        while lines and not lines[-1].strip():   # tmux pads the visible area with blank rows
            lines.pop()
        return lines
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[warn] log capture failed: {e!r}\n")
        return None


def new_lines(pre, post):
    """Lines that appeared in `post` after the end of `pre` (pane snapshots)."""
    if post is None:
        return []
    pre = list(pre) if pre else []
    post = list(post)
    while pre and not pre[-1].strip():
        pre.pop()
    while post and not post[-1].strip():
        post.pop()
    if not pre:
        return post
    for k in (8, 4, 2, 1):
        if len(pre) < k:
            continue
        tail = pre[-k:]
        for i in range(len(post) - k, -1, -1):
            if post[i:i + k] == tail:
                return post[i + k:]
    sys.stderr.write("[warn] could not align pane snapshots (scrollback overflow?) — using full post snapshot\n")
    return list(post)


def parse_engine_lines(lines, concurrency):
    gen = []
    spec = []
    for ln in lines:
        g = GEN_RE.search(ln)
        if g:
            r = RUN_RE.search(ln)
            p = PROMPT_RE.search(ln)
            gen.append((float(g.group(1)), int(r.group(1)) if r else -1, float(p.group(1)) if p else 0.0))
        if "SpecDecoding metrics" in ln:
            d = {}
            for key, rx in (("mal", SPEC_MAL_RE), ("accepted", SPEC_ACC_RE),
                            ("drafted", SPEC_DRF_RE), ("rate", SPEC_RATE_RE)):
                m = rx.search(ln)
                if m:
                    d[key] = float(m.group(1))
            m = SPEC_POS_RE.search(ln)
            if m:
                d["per_pos"] = [float(x) for x in m.group(1).split(",")]
            if d:
                spec.append(d)

    steady = [g for g, r, _ in gen if r >= concurrency and g > 0]
    active = [g for g, r, _ in gen if r >= 1 and g > 0]
    out = {
        "n_throughput_lines": len(gen),
        "n_steady_lines": len(steady),
        "server_gen_tps_steady_median": statistics.median(steady) if steady else None,
        "server_gen_tps_steady_min": min(steady) if steady else None,
        "server_gen_tps_steady_max": max(steady) if steady else None,
        "server_gen_tps_active_median": statistics.median(active) if active else None,
        "n_spec_lines": len(spec),
    }
    accs = [d["accepted"] for d in spec if "accepted" in d]
    drfs = [d["drafted"] for d in spec if "drafted" in d]
    # vLLM's SpecDecodingLogging resets its counters after every log line (per-interval).
    # Guard anyway: if counters look cumulative (monotone over >=3 lines), diff them.
    cumulative = (len(accs) >= 3 and len(accs) == len(drfs)
                  and all(a2 >= a1 for a1, a2 in zip(accs, accs[1:]))
                  and all(b2 >= b1 for b1, b2 in zip(drfs, drfs[1:]))
                  and drfs[-1] > 5 * max(drfs[0], 1))
    if cumulative:
        acc, drf = accs[-1] - accs[0], drfs[-1] - drfs[0]
    else:
        acc, drf = sum(accs), sum(drfs)
    rates = [d["rate"] for d in spec if "rate" in d]
    mals = [d["mal"] for d in spec if "mal" in d]
    pos = [d["per_pos"] for d in spec if "per_pos" in d]
    per_pos_mean = None
    if pos:
        w = min(len(p) for p in pos)
        per_pos_mean = [round(statistics.mean(p[i] for p in pos), 4) for i in range(w)]
    out.update({
        "mtp_accept_rate_token_weighted_pct": round(100.0 * acc / drf, 2) if drf else None,
        "mtp_accept_rate_line_median_pct": statistics.median(rates) if rates else None,
        "mtp_mean_acceptance_length_median": statistics.median(mals) if mals else None,
        "mtp_per_position_mean": per_pos_mean,
        "mtp_accepted_tokens": acc,
        "mtp_drafted_tokens": drf,
        "mtp_counts_treated_as_cumulative": cumulative,
    })
    return out



# --------------------------------------------------------------------------- /metrics (engine counters)
# Primary engine-side source. vLLM's Prometheus endpoint exposes the same counters its
# periodic log lines are printed from; sampling it every --metrics-interval seconds during a
# run gives steady-state throughput (windows with running >= concurrency) and exact
# token-weighted MTP acceptance, without depending on where the server logs.
METRIC_SUFFIX = {
    "gen": "generation_tokens_total",
    "prompt": "prompt_tokens_total",
    "drafts": "spec_decode_num_drafts_total",
    "draft_tokens": "spec_decode_num_draft_tokens_total",
    "accepted": "spec_decode_num_accepted_tokens_total",
    "per_pos": "spec_decode_num_accepted_tokens_per_pos_total",
    "running": "num_requests_running",
}
METRIC_LINE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[-+]Inf)\s*$")
POS_RE = re.compile(r'position="(\d+)"')


def fetch_text(url, timeout=10):
    try:
        with urlreq.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None


def parse_metrics(text):
    """Sum each counter over all label sets (model_name, engine, ...); per-position kept by position."""
    out = {"per_pos": {}}
    if not text:
        return out
    for ln in text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        m = METRIC_LINE_RE.match(ln)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        for key, suffix in METRIC_SUFFIX.items():
            if name.endswith(suffix):
                if key == "per_pos":
                    pm = POS_RE.search(labels)
                    if pm:
                        pos = int(pm.group(1))
                        out["per_pos"][pos] = out["per_pos"].get(pos, 0.0) + v
                else:
                    out[key] = out.get(key, 0.0) + v
                break
    return out


class MetricsSampler(threading.Thread):
    def __init__(self, url, interval):
        super().__init__(daemon=True)
        self.url, self.interval = url, interval
        self.samples = []          # (t_perf, parsed)
        self.raw_first = self.raw_last = None
        self._halt = threading.Event()

    def sample_once(self):
        txt = fetch_text(self.url)
        if txt:
            if self.raw_first is None:
                self.raw_first = txt
            self.raw_last = txt
            self.samples.append((time.perf_counter(), parse_metrics(txt)))

    def run(self):
        while not self._halt.is_set():
            self.sample_once()
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=30)
        self.sample_once()


def engine_from_samples(samples, concurrency):
    """Run-level engine metrics from /metrics samples: deltas of cumulative counters."""
    if len(samples) < 2:
        return {"metrics_n_samples": len(samples)}
    (t0, a), (t1, b) = samples[0], samples[-1]

    def d(key):
        if key in a and key in b:
            return b[key] - a[key]
        return None

    rates = []
    for (ta, pa), (tb, pb) in zip(samples, samples[1:]):
        if "gen" in pa and "gen" in pb and tb > ta:
            rates.append(((pb["gen"] - pa["gen"]) / (tb - ta), pb.get("running", -1)))
    steady = [r for r, run in rates if run >= concurrency and r > 0]
    active = [r for r, run in rates if run >= 1 and r > 0]
    gen, drafts, dtoks, acc = d("gen"), d("drafts"), d("draft_tokens"), d("accepted")
    per_pos = None
    if drafts and a.get("per_pos") and b.get("per_pos"):
        keys = sorted(set(a["per_pos"]) & set(b["per_pos"]))
        per_pos = [round((b["per_pos"][k] - a["per_pos"][k]) / drafts, 4) for k in keys]
    return {
        "metrics_n_samples": len(samples),
        "metrics_window_s": round(t1 - t0, 1),
        "metrics_gen_tokens_delta": gen,
        "server_gen_tps_run_aggregate": round(gen / (t1 - t0), 2) if (gen and t1 > t0) else None,
        "server_gen_tps_steady_median": round(statistics.median(steady), 2) if steady else None,
        "server_gen_tps_steady_min": round(min(steady), 2) if steady else None,
        "server_gen_tps_steady_max": round(max(steady), 2) if steady else None,
        "server_gen_tps_active_median": round(statistics.median(active), 2) if active else None,
        "n_steady_windows": len(steady),
        "mtp_accept_rate_token_weighted_pct": round(100.0 * acc / dtoks, 2) if (dtoks and acc is not None) else None,
        "mtp_mean_acceptance_length_median": round(1.0 + acc / drafts, 3) if (drafts and acc is not None) else None,
        "mtp_per_position_mean": per_pos,
        "mtp_accepted_tokens": acc,
        "mtp_drafted_tokens": dtoks,
        "mtp_num_drafts": drafts,
        "engine_source": "metrics",
    }

# --------------------------------------------------------------------------- requests
def build_body(model, question, max_tokens, effort):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(SAMPLING)
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    return body


def chat_stream(base_url, body, timeout):
    data = json.dumps(body).encode()
    req = urlreq.Request(base_url + "/chat/completions", data=data,
                         headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.perf_counter()
    t_first = t_last = None
    reasoning, content = [], []
    finish, usage, n_chunks = None, None, 0
    with urlreq.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                r = d.get("reasoning_content") or d.get("reasoning")
                c = d.get("content")
                if r or c:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_chunks += 1
                if r:
                    reasoning.append(r)
                if c:
                    content.append(c)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    ctoks = (usage or {}).get("completion_tokens")
    ptoks = (usage or {}).get("prompt_tokens")
    ttft = (t_first - t0) if t_first else None
    decode_s = (t_last - t_first) if (t_first and t_last) else None
    decode_tps = ((ctoks - 1) / decode_s) if (ctoks and decode_s and decode_s > 0 and ctoks > 1) else None
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "finish_reason": finish,
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "latency_s": t_end - t0,
        "decode_tps": decode_tps,
        "e2e_tps": (ctoks / (t_end - t0)) if ctoks else None,
        "n_stream_chunks": n_chunks,
        "usage": usage,
    }


def http_json(url, timeout=10):
    try:
        with urlreq.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


# --------------------------------------------------------------------------- runs
def load_items(path, offset, n, grade=True):
    rows = []
    with open(path) as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            q = r.get("question") or r.get("prompt")
            gold = gold_answer(r["answer"]) if (grade and r.get("answer") is not None) else None
            rows.append({"qid": i, "question": q, "gold": gold})
    return rows[offset:offset + n] if n else rows[offset:]


def run_batch(args, items, tag):
    results = [None] * len(items)
    lock = threading.Lock()
    done = [0]
    t_start = time.perf_counter()

    def work(i):
        body = build_body(args.model, items[i]["question"], args.max_tokens, args.effort)
        try:
            r = chat_stream(args.base_url, body, args.timeout)
        except Exception as e:  # noqa: BLE001
            r = {"error": repr(e), "content": "", "reasoning": ""}
        r["qid"] = items[i]["qid"]
        r["gold"] = items[i]["gold"]
        r["question"] = items[i]["question"]
        if items[i]["gold"] is not None:
            r["pred"] = extract_answer(r.get("content") or "")
            r["correct"] = (r["pred"] == items[i]["gold"])
        return i, r, body

    first_body = None
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, i) for i in range(len(items))]
        for f in as_completed(futs):
            i, r, body = f.result()
            results[i] = r
            with lock:
                if first_body is None or i == 0:
                    first_body = body
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(items):
                    el = time.perf_counter() - t_start
                    sys.stderr.write(f"  [{tag}] {done[0]}/{len(items)} done, {el:.0f}s elapsed\n")
    wall = time.perf_counter() - t_start
    return results, wall, first_body


def summarize_run(results, wall, engine, args, run_idx):
    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]
    graded = [r for r in ok if r.get("gold") is not None]
    ctoks = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    dec = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    e2e = [r["e2e_tps"] for r in ok if r.get("e2e_tps")]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    lat = [r["latency_s"] for r in ok if r.get("latency_s") is not None]

    def med(xs):
        return statistics.median(xs) if xs else None

    def pct(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    s = {
        "run": run_idx,
        "label": args.label,
        "model": args.model,
        "n_requested": len(results),
        "n_ok": len(ok),
        "n_errors": len(err),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "effort": args.effort or "template-default",
        "sampling": SAMPLING,
        "wall_s": round(wall, 2),
        "em_correct": sum(1 for r in graded if r.get("correct")) if graded else None,
        "em_n": len(graded) if graded else None,
        "em_pct": round(100.0 * sum(1 for r in graded if r.get("correct")) / len(graded), 2) if graded else None,
        "truncated": sum(1 for r in ok if r.get("finish_reason") == "length"),
        "mean_completion_tokens": round(statistics.mean(ctoks), 1) if ctoks else None,
        "median_completion_tokens": med(ctoks),
        "total_completion_tokens": sum(ctoks),
        "mean_reasoning_chars": round(statistics.mean(len(r.get("reasoning") or "") for r in ok), 1) if ok else None,
        "mean_content_chars": round(statistics.mean(len(r.get("content") or "") for r in ok), 1) if ok else None,
        "median_decode_tps": round(med(dec), 2) if dec else None,
        "p10_decode_tps": round(pct(dec, 0.10), 2) if dec else None,
        "p90_decode_tps": round(pct(dec, 0.90), 2) if dec else None,
        "median_e2e_tps": round(med(e2e), 2) if e2e else None,
        "median_ttft_s": round(med(ttft), 3) if ttft else None,
        "median_latency_s": round(med(lat), 2) if lat else None,
        "p90_latency_s": round(pct(lat, 0.90), 2) if lat else None,
        "client_aggregate_tps": round(sum(ctoks) / wall, 2) if (ctoks and wall > 0) else None,
    }
    s.update(engine)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="ukisai/Swift-Qwen3.8-27B-NVFP4")
    ap.add_argument("--label", default="swift")
    ap.add_argument("--out", default=None, help="results dir (default results/<label>/conc<N>)")
    ap.add_argument("--data", default="data/gsm8k_test.jsonl")
    ap.add_argument("--prompts", default=None, help="alternative JSONL (question[,answer]) instead of GSM8K")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=8, help="unmeasured requests before EACH run, taken from beyond offset+n")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--effort", default=None, help="reasoning_effort chat_template_kwarg (low/medium/high/xhigh); default: none = template default")
    ap.add_argument("--log-cmd", default="tmux capture-pane -t cluster -p -J -S -",
                    help="shell command that prints the vLLM engine log (the tmux pane). '' to disable.")
    ap.add_argument("--metrics-url", default=None,
                    help="vLLM Prometheus endpoint (default: <base-url minus /v1>/metrics). '' to disable.")
    ap.add_argument("--metrics-interval", type=float, default=10.0, help="seconds between /metrics samples during a run")
    ap.add_argument("--settle", type=float, default=12.0, help="seconds to wait for the last engine log line before snapshotting")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    out = args.out or os.path.join("results", args.label, f"conc{args.concurrency}")
    metrics_url = args.metrics_url if args.metrics_url is not None else args.base_url.rsplit("/v1", 1)[0] + "/metrics"
    if metrics_url and not fetch_text(metrics_url):
        sys.stderr.write(f"[warn] /metrics not reachable at {metrics_url}; engine-side numbers will come from the log pane only\n")
        metrics_url = ""
    os.makedirs(out, exist_ok=True)
    src = args.prompts or args.data
    items = load_items(src, args.offset, args.n, grade=not args.no_grade)
    warm = load_items(src, args.offset + args.n, args.warmup, grade=False) if args.warmup else []
    if args.warmup and len(warm) < args.warmup:
        sys.stderr.write(f"[warn] only {len(warm)} warmup items available beyond offset+n\n")
    if not items:
        sys.exit("no items loaded")

    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "args": vars(args),
        "sampling": SAMPLING,
        "answer_extraction": EXTRACTION_RULE,
        "server_version": http_json(args.base_url.rsplit("/v1", 1)[0] + "/version"),
        "metrics_url": metrics_url,
        "metrics_available": bool(metrics_url and fetch_text(metrics_url)),
        "server_models": http_json(args.base_url + "/models"),
        "data_source": src,
        "n_items": len(items),
        "qid_range": [items[0]["qid"], items[-1]["qid"]],
    }
    with open(os.path.join(out, "run-meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    sys.stderr.write(f"== {args.label}: {args.model} | n={len(items)} conc={args.concurrency} runs={args.runs} "
                     f"warmup={len(warm)} max_tokens={args.max_tokens} effort={args.effort or 'template-default'}\n")

    summaries = []
    for k in range(1, args.runs + 1):
        if warm:
            sys.stderr.write(f"-- run {k}: warmup ({len(warm)} reqs, unmeasured)\n")
            run_batch(args, warm, f"warm{k}")
            time.sleep(args.settle)
        pre = capture_log(args.log_cmd)
        sampler = None
        if metrics_url:
            sampler = MetricsSampler(metrics_url, args.metrics_interval)
            sampler.sample_once()
            sampler.start()
        t_wall = datetime.now().isoformat(timespec="seconds")
        sys.stderr.write(f"-- run {k}: measured ({len(items)} reqs @ concurrency {args.concurrency})\n")
        results, wall, first_body = run_batch(args, items, f"run{k}")
        time.sleep(args.settle)
        if sampler:
            sampler.stop()
        post = capture_log(args.log_cmd)
        lines = new_lines(pre, post)
        with open(os.path.join(out, f"run{k}.vllm-log.txt"), "w") as f:
            f.write(f"# engine log lines captured between {t_wall} and {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"# log-cmd: {args.log_cmd}\n")
            f.write("\n".join(lines) + "\n")
        engine = parse_engine_lines(lines, args.concurrency)
        engine["engine_source"] = "log" if engine["n_throughput_lines"] else None
        if sampler:
            for tag, raw in (("pre", sampler.raw_first), ("post", sampler.raw_last)):
                if raw:
                    with open(os.path.join(out, f"run{k}.metrics-{tag}.txt"), "w") as f:
                        f.write(raw)
            with open(os.path.join(out, f"run{k}.metrics-samples.jsonl"), "w") as f:
                for t, pm in sampler.samples:
                    f.write(json.dumps({"t": round(t, 3), **{kk: vv for kk, vv in pm.items()}}) + "\n")
            m = engine_from_samples(sampler.samples, args.concurrency)
            if m.get("metrics_gen_tokens_delta"):
                engine.update({kk: vv for kk, vv in m.items() if vv is not None})   # /metrics is primary; never overwrite with None
            else:
                sys.stderr.write("[warn] /metrics samples carried no generation-token delta (metric names differ?) — see run{k}.metrics-post.txt\n")
        if engine.get("server_gen_tps_steady_median") is None:
            sys.stderr.write("[warn] no engine-side throughput for this run (no /metrics delta and no log lines)\n")
        if engine.get("mtp_accept_rate_token_weighted_pct") is None:
            sys.stderr.write("[warn] no MTP acceptance for this run — is speculative decoding engaged?\n")
        with open(os.path.join(out, f"run{k}.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if k == 1 and first_body:
            with open(os.path.join(out, "request-sample.json"), "w") as f:
                json.dump(first_body, f, indent=2, ensure_ascii=False)
        s = summarize_run(results, wall, engine, args, k)
        s["started"] = t_wall
        summaries.append(s)
        with open(os.path.join(out, f"run{k}.summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        sys.stderr.write(
            f"   run {k}: EM={s['em_pct']}% trunc={s['truncated']} mean_ctok={s['mean_completion_tokens']} "
            f"med_decode_tps={s['median_decode_tps']} server_steady_tps={s['server_gen_tps_steady_median']} "
            f"mtp_acc={s['mtp_accept_rate_token_weighted_pct']}% MAL={s['mtp_mean_acceptance_length_median']} "
            f"src={s.get('engine_source')} "
            f"wall={s['wall_s']}s errors={s['n_errors']}\n")
    sys.stderr.write(f"done -> {out}  (run `report.py summarize` for median/min/max)\n")
    return 0


# --------------------------------------------------------------------------- selftest
SAMPLE_LOG = """
INFO 09-13 10:00:01 [loggers.py:123] Engine 000: Avg prompt throughput: 812.3 tokens/s, Avg generation throughput: 21.4 tokens/s, Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%
INFO 09-13 10:00:01 [metrics.py:99] SpecDecoding metrics: Mean acceptance length: 2.31, Accepted throughput: 45.20 tokens/s, Drafted throughput: 98.10 tokens/s, Accepted: 452 tokens, Drafted: 981 tokens, Per-position acceptance rate: 0.670, 0.430, 0.220, Avg Draft acceptance rate: 46.1%
INFO 09-13 10:00:11 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 96.7 tokens/s, Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.0%, Prefix cache hit rate: 0.0%
INFO 09-13 10:00:11 [metrics.py:99] SpecDecoding metrics: Mean acceptance length: 2.40, Accepted throughput: 60.00 tokens/s, Drafted throughput: 120.00 tokens/s, Accepted: 600 tokens, Drafted: 1200 tokens, Per-position acceptance rate: 0.700, 0.450, 0.250, Avg Draft acceptance rate: 50.0%
INFO 09-13 10:00:21 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 98.1 tokens/s, Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.1%, Prefix cache hit rate: 0.0%
INFO 09-13 10:00:31 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 40.2 tokens/s, Running: 3 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 0.0%
INFO 09-13 10:00:41 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
"""


SAMPLE_METRICS_A = """# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="m"} 1000.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 500.0
vllm:num_requests_running{engine="0",model_name="m"} 8.0
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 300.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 150.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 80.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 45.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="2"} 25.0
vllm:request_generation_tokens_bucket{le="10"} 3.0
"""
SAMPLE_METRICS_B = SAMPLE_METRICS_A.replace("1000.0", "2000.0").replace("} 100.0", "} 300.0").replace("} 300.0\nvllm:spec_decode_num_accepted", "} 900.0\nvllm:spec_decode_num_accepted") \
    .replace("} 150.0", "} 450.0").replace("} 80.0", "} 240.0").replace("} 45.0", "} 135.0").replace("} 25.0", "} 75.0")

def selftest():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'} {name}: got={got!r} want={want!r}")

    print("answer extraction:")
    check("hash", extract_answer("So 3+4 = 7.\n#### 7"), "7")
    check("hash-money", extract_answer("Total is $1,250.\n#### $1,250"), "1250")
    check("boxed", extract_answer("Therefore \\boxed{18} eggs."), "18")
    check("boxed-decimal", extract_answer("\\boxed{2.5}"), "2.5")
    check("last-number", extract_answer("She has 16 eggs, sells 7, keeps 9."), "9")
    check("comma", extract_answer("The answer is 1,000,000."), "1000000")
    check("empty", extract_answer(""), None)
    # cases from the audit of the first Swift pass (v1 got every one of these wrong)
    check("bold-then-explanation", extract_answer("It takes **3 bolts in total**: 2 bolts of blue fiber + 1 bolt of white fiber."), "3")
    check("latex-thousands", extract_answer("His profit is 200{,}000 - 130{,}000 = 70{,}000.\n**Answer: $\\$70{,}000$**"), "70000")
    check("bold-first-on-line", extract_answer("So it takes **160 minutes**, or **2 hours 40 minutes**."), "160")
    check("bold-with-trailing", extract_answer("So, she will eat **7 dozen eggs** in 4 weeks."), "7")
    check("bold-sentence", extract_answer("**Judy makes $7,425 in 1 week.**"), "7425")
    check("latex-money-bold", extract_answer("**Answer: \\(\\$9{,}360\\)**"), "9360")
    check("boxed-text", extract_answer("\\[ \\boxed{36.4\\text{ seconds}} \\]"), "36.4")
    check("hyphen-not-minus", extract_answer("(since 80-150-170 is a right triangle)."), "170")
    check("step-header-skipped", extract_answer("**Step 3:** 12 x 4 = **48 quilt blocks**."), "48")
    check("thin-space", extract_answer("Total 7\\,200 dollars"), "7200")
    check("v1 reference", extract_answer_v1("**7 dozen eggs** in 4 weeks."), "4")
    # v3: equations and money inside the chosen segment (from the v2 residual-miss review)
    check("bold-equation", extract_answer("Total: **$500 + $800 + $130 = $1,430**"), "1430")
    check("bold-equation-2", extract_answer("Total kittens: **7 + 21 + 12 = 40**"), "40")
    check("bold-money-sentence", extract_answer("**8 pens will cost $12.00.**"), "12")
    check("bold-money-years", extract_answer("**Her annual pension after quitting at 30 years would be $25,000/year.**"), "25000")
    check("prefer-eq-segment-on-line", extract_answer("A robe takes **2 bolts of blue fiber** and **1 bolt** of white. Total: **2 + 1 = 3 bolts**."), "3")
    check("boxed-equation", extract_answer("\\boxed{x = 42}"), "42")
    check("answer-equation", extract_answer("Answer: 100 + 6 = 106 dollars"), "106")
    check("plain-bold-still-first", extract_answer("So it takes **160 minutes**, or **2 hours 40 minutes**."), "160")
    check("gold", gold_answer("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n#### 18"), "18")

    print("engine log parsing (concurrency 8):")
    e = parse_engine_lines(SAMPLE_LOG.strip().splitlines(), 8)
    check("throughput lines", e["n_throughput_lines"], 5)
    check("steady lines (Running>=8)", e["n_steady_lines"], 3)
    check("steady median", e["server_gen_tps_steady_median"], 96.7)
    check("spec lines", e["n_spec_lines"], 2)
    check("token-weighted MTP %", e["mtp_accept_rate_token_weighted_pct"], round(100 * (452 + 600) / (981 + 1200), 2))
    check("MAL median", e["mtp_mean_acceptance_length_median"], statistics.median([2.31, 2.40]))
    check("per-position mean", e["mtp_per_position_mean"], [0.685, 0.44, 0.235])
    check("not cumulative", e["mtp_counts_treated_as_cumulative"], False)

    print("/metrics parsing:")
    a, b = parse_metrics(SAMPLE_METRICS_A), parse_metrics(SAMPLE_METRICS_B)
    check("gen counter", a["gen"], 1000.0)
    check("histogram ignored", "request_generation" in str(a), False)
    check("per-pos parsed", a["per_pos"], {0: 80.0, 1: 45.0, 2: 25.0})
    e = engine_from_samples([(0.0, a), (10.0, b)], 8)
    check("gen delta", e["metrics_gen_tokens_delta"], 1000.0)
    check("aggregate tps", e["server_gen_tps_run_aggregate"], 100.0)
    check("steady median", e["server_gen_tps_steady_median"], 100.0)
    check("MTP % (300 acc / 600 drafted)", e["mtp_accept_rate_token_weighted_pct"], 50.0)
    check("MAL (1 + 300/200)", e["mtp_mean_acceptance_length_median"], 2.5)
    check("per-pos", e["mtp_per_position_mean"], [0.8, 0.45, 0.25])
    print("snapshot diff:")
    pre = ["a", "b", "c", "d"]
    post = ["x", "b", "c", "d", "e", "f"]
    check("new lines", new_lines(pre, post), ["e", "f"])
    check("no pre", new_lines(None, post), post)
    check("trailing blank rows", new_lines(["a", "b", "c", "d", ""], ["b", "c", "d", "e", "f", ""]), ["e", "f"])

    print("request body:")
    b = build_body("m", "q", 8192, None)
    check("sampling", {k: b[k] for k in SAMPLING}, SAMPLING)
    check("no effort kwarg", "chat_template_kwargs" in b, False)
    check("effort kwarg", build_body("m", "q", 1, "xhigh")["chat_template_kwargs"], {"reasoning_effort": "xhigh"})
    print(f"selftest: {'OK' if fails == 0 else str(fails) + ' FAILED'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
