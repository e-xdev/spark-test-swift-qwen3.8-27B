#!/usr/bin/env python3
"""
mock_vllm.py — a tiny stand-in for vLLM's OpenAI-compatible server so the harness,
report and compare tooling can be exercised on any machine (no GPU, no network).

  python3 tests/mock_vllm.py --port 8123 --model mock/model --accuracy 0.8 &
  python3 tests/mock_log.py reset
  python3 scripts/gsm8k-bench.py --base-url http://localhost:8123/v1 --model mock/model \
      --data tests/mini_gsm8k.jsonl --n 8 --concurrency 4 --runs 2 --warmup 2 \
      --log-cmd "python3 tests/mock_log.py" --settle 0.2 --out /tmp/mock/swift/conc8

It streams reasoning_content deltas, then content deltas, then a usage-only chunk, then
[DONE] — the same shape vLLM emits with --reasoning-parser qwen3 and stream_options.include_usage.
The final answer is right with probability --accuracy (deterministic per (qid, run) via the
request counter) and is written as '\\boxed{N}' or '#### N' alternately to test extraction.
"""
import argparse
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
COUNTER = [0]
LOCK = threading.Lock()
METRICS = {"gen": 0.0, "prompt": 0.0, "drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "pos": [0.0, 0.0, 0.0], "running": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.endswith("/models"):
            return self._json({"object": "list", "data": [{"id": ARGS.model, "object": "model"},
                                                          {"id": "alias-" + ARGS.model.split("/")[-1], "object": "model"}]})
        if self.path.endswith("/version"):
            return self._json({"version": "mock-0.0"})
        if self.path.endswith("/metrics"):
            with LOCK:
                m = dict(METRICS); pos = list(METRICS["pos"])
            lab = '{engine="0",model_name="%s"}' % ARGS.model
            body = "\n".join([
                "# HELP vllm:generation_tokens_total Number of generation tokens processed.",
                "# TYPE vllm:generation_tokens_total counter",
                "vllm:generation_tokens_total%s %.1f" % (lab, m["gen"]),
                "vllm:prompt_tokens_total%s %.1f" % (lab, m["prompt"]),
                "vllm:num_requests_running%s %.1f" % (lab, m["running"]),
                "vllm:spec_decode_num_drafts_total%s %.1f" % (lab, m["drafts"]),
                "vllm:spec_decode_num_draft_tokens_total%s %.1f" % (lab, m["draft_tokens"]),
                "vllm:spec_decode_num_accepted_tokens_total%s %.1f" % (lab, m["accepted"]),
            ] + ['vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="%s",position="%d"} %.1f' % (ARGS.model, i, v) for i, v in enumerate(pos)]) + "\n"
            b = body.encode()
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b); return
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        assert body.get("temperature") == 1.0 and body.get("top_k") == 20, "sampling not forwarded"
        q = body["messages"][-1]["content"]
        with LOCK:
            COUNTER[0] += 1
            c = COUNTER[0]
            METRICS["running"] += 1
        rng = random.Random(c)
        nums = re.findall(r"\d+", q)
        gold = None
        m = re.search(r"\[gold=(\d+)\]", q)
        if m:
            gold = int(m.group(1))
        ans = gold if (gold is not None and rng.random() < ARGS.accuracy) else (gold or 0) + 1
        style = "\\boxed{%d}" % ans if c % 2 else "#### %d" % ans
        reasoning = ["Let me think. ", "The numbers are %s. " % ", ".join(nums[:3]), "So the result is %d. " % ans]
        content = ["The answer is ", style + "."]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def chunk(delta, finish=None):
            return ("data: " + json.dumps({"id": "c%d" % c, "object": "chat.completion.chunk", "model": ARGS.model,
                                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()

        time.sleep(ARGS.ttft)
        self.wfile.write(chunk({"role": "assistant", "content": ""}))
        for r in reasoning:
            self.wfile.write(chunk({"reasoning_content": r})); self.wfile.flush(); time.sleep(ARGS.itl)
        for t in content[:-1]:
            self.wfile.write(chunk({"content": t})); self.wfile.flush(); time.sleep(ARGS.itl)
        self.wfile.write(chunk({"content": content[-1]}, finish="stop")); self.wfile.flush()
        toks = 20 + len(nums) * 3 + (c % 7)
        self.wfile.write(("data: " + json.dumps({"id": "c%d" % c, "object": "chat.completion.chunk", "choices": [],
                                                  "usage": {"prompt_tokens": 40, "completion_tokens": toks,
                                                            "total_tokens": 40 + toks}}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        with LOCK:  # engine counters, like vLLM: every generated token was proposed by the MTP head in drafts of 3
            d = toks // 3 + 1
            METRICS["gen"] += toks; METRICS["prompt"] += 40; METRICS["drafts"] += d; METRICS["draft_tokens"] += 3 * d
            acc = [round(d * r) for r in (0.7, 0.45, 0.25)]
            METRICS["accepted"] += sum(acc)
            for i, v in enumerate(acc): METRICS["pos"][i] += v
            METRICS["running"] -= 1


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--model", default="mock/model")
    ap.add_argument("--accuracy", type=float, default=0.8)
    ap.add_argument("--ttft", type=float, default=0.05)
    ap.add_argument("--itl", type=float, default=0.03)
    ARGS = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), H)
    print(f"mock vLLM on :{ARGS.port} model={ARGS.model} accuracy={ARGS.accuracy}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
