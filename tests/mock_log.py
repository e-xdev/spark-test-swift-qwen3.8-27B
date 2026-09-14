#!/usr/bin/env python3
"""Stateful fake of `tmux capture-pane -p -S -`: every call appends 3 engine lines and prints the whole pane."""
import os, random, sys
STATE = "/tmp/mock_vllm_pane.txt"
if len(sys.argv) > 1 and sys.argv[1] == "reset":
    open(STATE, "w").close(); sys.exit(0)
lines = open(STATE).read().splitlines() if os.path.exists(STATE) else []
n = len(lines) // 3
rng = random.Random(n)
for i in range(3):
    t = n * 3 + i
    running = 8 if i < 2 else 3
    gen = rng.uniform(90, 100) if running == 8 else rng.uniform(30, 40)
    lines.append(f"INFO 09-13 10:{t//60:02d}:{t%60:02d} [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, "
                 f"Avg generation throughput: {gen:.1f} tokens/s, Running: {running} reqs, Waiting: 0 reqs, GPU KV cache usage: 1.0%, Prefix cache hit rate: 0.0%")
    acc, drf = rng.randint(400, 600), rng.randint(1000, 1200)
    lines.append(f"INFO 09-13 10:{t//60:02d}:{t%60:02d} [metrics.py:99] SpecDecoding metrics: Mean acceptance length: {1+acc/(drf/3):.2f}, "
                 f"Accepted throughput: {acc/10:.2f} tokens/s, Drafted throughput: {drf/10:.2f} tokens/s, Accepted: {acc} tokens, Drafted: {drf} tokens, "
                 f"Per-position acceptance rate: 0.670, 0.430, 0.220, Avg Draft acceptance rate: {100*acc/drf:.1f}%")
    lines.append(f"INFO 09-13 10:{t//60:02d}:{t%60:02d} [async_llm.py:1] Added request chatcmpl-{t}.")
open(STATE, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
