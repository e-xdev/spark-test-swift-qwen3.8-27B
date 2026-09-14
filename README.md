# Swift-Qwen3.8-27B-NVFP4 on NVIDIA DGX Spark (GB10 / SM121) — reproducible benchmark kit

Reproducible throughput, MTP-acceptance and GSM8K exact-match numbers for
[`ukisai/Swift-Qwen3.8-27B-NVFP4`](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4)
on a single DGX Spark, mirroring the protocol on the model card so the rows line up with
UkisAI's RTX PRO 6000 (SM120) column — plus the same protocol run on the base
`unsloth/Qwen3.8-27B-NVFP4` on the same box, so Swift's token savings can be read as
time-to-answer on the same hardware.

**Results: see [REPORT.md](REPORT.md).** Headline, 8 concurrent, GSM8K-200 × 5 runs, medians:

| | Swift-NVFP4 — DGX Spark (this repo) | Swift-NVFP4 — RTX PRO 6000 (card) | base Qwen3.8-27B-NVFP4 (unsloth) — same Spark |
|---|---|---|---|
| Exact match | **97.0 %** (95.0–98.0) | 96.5 % | 96.0 % (95.0–97.5) |
| Median tok/s per request | **16.4** | 85 | 20.0 |
| Engine aggregate tok/s | 118 | — | 144 |
| Mean completion tokens | **340** | 391 | 427 |
| Wall time, 200 questions | **598 s** | — | 699 s |
| MTP draft acceptance | **66.0 %** | 61 % | 62.2 % |
| Kernel | native `FlashInferCutlassNvFp4LinearKernel` (sm_121a) | native (SM120) | same as Swift |

Raw per-run outputs, engine-counter snapshots and startup logs are under `results/`.

## Why this exists

UkisAI benchmarked NVFP4 on an RTX PRO 6000 Blackwell (SM120). DGX Spark's GB10 is SM121, a part
their card does not mention. Older reports for GB10 needed a Marlin weight-only fallback
(`VLLM_NVFP4_GEMM_BACKEND=marlin`) to avoid `!!!!!` garbage. **On the build used here that is no
longer the case:** the startup log shows

```
Unknown vLLM environment variable detected: VLLM_NVFP4_GEMM_BACKEND
Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM     (FlashInfer 0.6.15, sm_121a kernels)
```

so the numbers here are the **native FP4 path on GB10**, and the remaining gap to the card's SM120
column is dominated by memory bandwidth (273 GB/s vs ~1.8 TB/s), not by a dequant kernel. The
inert flags are kept in the recipe for older images and documented as such.

## Protocol (mirrors the model card)

| | value |
|---|---|
| Benchmark | GSM8K test split, **first 200 questions**, exact match |
| Sampling | temperature 1.0 · top_p 0.95 · top_k 20 · min_p 0 (sent on every request; first body saved) |
| Concurrency | **8** (constant 8 in flight over the 200 questions) — plus a clearly labeled single-stream row |
| Output cap | max_tokens 8192; truncations counted |
| Reasoning effort | template default (no `reasoning_effort` kwarg), as on the NVFP4 card. `EFFORT=xhigh` to change. |
| MTP | `--speculative-config.method mtp`, 3 draft tokens (in the recipe) |
| Runs | **5 measured runs**, each preceded by 8 unmeasured warmup requests (GSM8K rows 200–207); report median (min–max) |
| Speed | engine-side: vLLM's `/metrics` counters sampled every 10 s during the run — steady state = windows with `running >= concurrency`; client-side: per-request **decode-phase** tok/s (first→last streamed token, TTFT excluded) |
| MTP stats | engine-side `/metrics` spec-decode counters, token-weighted deltas over the run (accepted / drafted; mean acceptance length = 1 + accepted / drafts; per-position rates) |
| Exact match | extraction rule v3 on the final (non-reasoning) answer: `#### N` → last `\boxed{}` → text after the last "answer is / Answer:" → the bold segment on the last bolded line (one containing `=` preferred) → last number; inside the chosen segment: number after the last `=` → last `$` amount → first number. LaTeX `{,}` / `\,` separators removed, `$` and commas stripped, numeric compare, a `-` after a word character is not a minus. `scripts/regrade.py` applies the current rule to saved outputs, so both models are graded identically; REPORT.md §3 classifies the residual misses |

Where the card is silent (harness, N runs behind "median", prompt formatting) this kit makes
a choice and states it; the question is sent as the bare user message with no system prompt.

## Layout

```
recipes/swift-qwen38-nvfp4.yaml        # known-good eugr/spark-vllm-docker recipe (env flags + MTP-3)
recipes/qwen38-27b-nvfp4-base.yaml     # identical, base model
scripts/prepare-gsm8k.sh               # fetch GSM8K test split, sanity-check, sha256
scripts/eval-swift-quant.sh            # the fixed eval wrapper: preflight | smoke | single | concurrent | report | all
scripts/gsm8k-bench.py                 # the harness (stdlib only); --selftest checks the parsers
scripts/report.py                      # summarize (median/min/max) and compare (side-by-side + paired outcomes)
scripts/regrade.py                     # re-grade saved outputs with the current extraction rule; lists residual misses
scripts/run-side-by-side.sh            # Swift then base, sequentially, pauses before every destructive step
prompts/smoke.jsonl                    # garbage/coherence smoke prompts (EN + ES); not an accuracy score
tests/                                 # mock vLLM server + fake engine log: validate the kit without a GPU
results/<label>/{smoke,single,conc8}/  # per-run JSONL (every output), summaries, engine-log lines, request sample
```

## One-time setup (on the DGX Spark)

```bash
git clone https://github.com/em-xdev/spark-test-swift-qwen3.8-27B ~/spark-test-swift-qwen3.8-27B && cd ~/spark-test-swift-qwen3.8-27B
python3 scripts/gsm8k-bench.py --selftest          # parsers OK?
tests/run-mock-test.sh                             # whole pipeline OK? (no GPU needed, ~1 min)
scripts/prepare-gsm8k.sh                           # data/gsm8k_test.jsonl (1319 rows) + sha256
cp recipes/*.yaml ~/spark-vllm-docker/recipes/     # then: cd ~/spark-vllm-docker && ./run-recipe.sh swift-qwen38-nvfp4 --solo --dry-run
hf download unsloth/Qwen3.8-27B-NVFP4              # ~29 GB, only needed for the side-by-side
```

**How the server logs.** `run-recipe.sh` runs `docker exec … vllm serve` in the foreground of the
shell that started it; `docker logs vllm_node` stays empty. Launch it inside a detached tmux session
with a large scrollback so the startup lines (kernel selection, launch args) survive as provenance:

```bash
tmux set-option -g history-limit 50000
tmux new -d -s cluster -c ~/spark-vllm-docker './run-recipe.sh swift-qwen38-nvfp4 --solo; exec bash'
until curl -sf localhost:8000/v1/models | grep -q ukisai/Swift; do sleep 10; done
mkdir -p results/swift && tmux capture-pane -t cluster -p -J -S - | grep -iE 'marlin|nvfp4|LinearKernel|non-default args|Unknown vLLM|specul' > results/swift/startup-lines.txt
```

Never press Ctrl-C in that pane (it stops the server); `Ctrl-b d` detaches. The benchmark itself
reads engine numbers from `/metrics`, so it works even if the log lands somewhere else.

## Running

Run the eval itself from its own tmux window (not the `cluster` one) — a full pass is hours.

```bash
# Swift, with the model already serving on :8000
LABEL=swift MODEL=ukisai/Swift-Qwen3.8-27B-NVFP4 scripts/eval-swift-quant.sh all
#   preflight  -> models.json, vllm-version.json, metrics-at-preflight.txt, kernel-path.txt, env.txt, nvidia-smi.txt
#   smoke      -> 5 prompts single-stream, garbage count (expect 0)
#   single     -> 20 questions x 5 runs @ concurrency 1
#   concurrent -> 200 questions x 5 runs @ concurrency 8   (the card row)
#   report     -> results/swift/summary.md

# Base model (after stopping Swift: docker rm -f vllm_node; launch the base recipe)
LABEL=base MODEL=unsloth/Qwen3.8-27B-NVFP4 scripts/eval-swift-quant.sh all

# Side by side, in the card's row format, with the card's SM120 column for reference
python3 scripts/report.py compare results/swift results/base --with-card --out results/compare.md
```

Or let `scripts/run-side-by-side.sh` drive both, sequentially (run it from a plain shell, not
inside tmux); it dry-runs each launch, asks before stopping or starting a server, and launches
each server inside a detached `cluster` tmux session. Never serve both models at once — they share the 273 GB/s
memory bus and every number would be meaningless.

Knobs (env): `RUNS WARMUP EFFORT N_CONC CONC N_SINGLE MAX_TOKENS SETTLE BASE_URL TMUX_TARGET LOG_CMD OUT DATA`
(`LOG_CMD=''` disables pane capture; engine numbers then come from `/metrics` alone).

Rough duration per model at ~390 completion tokens/question: concurrent 5×200 ≈ 1–2 h,
single 5×20 ≈ 45 min, smoke ≈ 2 min.

## Reading the numbers

* `Median tok/s per request, decode phase` — the analogue of the card's "median tokens/s per
  request (8 concurrent)". Measured client-side over first→last streamed token, so prefill and
  the HTTP round-trip are excluded. On localhost the remaining client overhead is negligible.
* `Engine aggregate generation tok/s, steady state` — delta of vLLM's `generation_tokens_total`
  per 10 s sampling window, median over windows where `num_requests_running >= concurrency`.
  Summed over all in-flight requests; per-request ≈ aggregate / 8. `…over the whole run` is the
  same counter over the full run wall time, including ramp-up and tail.
* `MTP draft-token acceptance` — Δaccepted / Δdraft_tokens over the run; mean acceptance length
  = 1 + Δaccepted / Δdrafts; per-position = Δaccepted_at_position / Δdrafts.
* `Paired outcome` — per question, run k of A against run k of B: both / only A / only B.
* Smoke test = garbage count and coherence only. It is not an accuracy percentage.
* Every run directory holds `runN.metrics-pre.txt`, `runN.metrics-post.txt` (raw `/metrics`
  snapshots), `runN.metrics-samples.jsonl` (the 10 s series), `runN.jsonl` (every prompt, reasoning,
  answer, timing) and `runN.vllm-log.txt` (pane lines, if any).

## Before publishing — checklist

- [ ] `results/*/startup-lines.txt` names the kernel actually used (`FlashInferCutlassNvFp4LinearKernel` on this build) and shows `speculative_config` with `method=mtp`, `num_speculative_tokens=3`.
- [ ] every `run*.summary.json` has `engine_source: metrics` and non-null `mtp_accept_rate_token_weighted_pct`.
- [ ] `results/*/conc8/request-sample.json` shows temp 1.0 / top_p 0.95 / top_k 20 / min_p 0 and no `reasoning_effort` (or the one you intend).
- [ ] `n_errors` = 0 in every `run*.summary.json`; truncation counts noted.
- [ ] `scripts/regrade.py results/<label>` run on every label with the same harness version, residual misses reviewed by hand and the model-error / grading-ambiguity split stated in REPORT.md (e.g. gold 36 vs answer 36.4).
- [ ] vLLM version (`vllm-version.json`) and image tag stated in REPORT.md; hardware line: DGX Spark GB10, 128 GB unified, 273 GB/s, SM121, single node.
- [ ] Single-stream and 8-concurrent rows are never mixed; the card's 85 tok/s is only ever placed next to the 8-concurrent row.
- [ ] The two env flags are described as inert on this build (see startup-lines.txt), not as required.
- [ ] Weights-on-disk and weights-in-GPU rows filled from `du -sh` and the vLLM "model weights take" line.
- [ ] Data sha256 from `data/gsm8k_test.sha256` recorded.
- [ ] `scripts/scrub-repo.py --user <login> --host <hostname> --handle <hf-user> --strip-ai` run before the first commit; its final scan says `clean`.

## Caveats (state them in REPORT.md)

Single node; 5 runs; one vLLM build (`0.25.1.dev24+g96bb89286.d20260710`, eugr/spark-vllm-docker
image, FlashInfer 0.6.15); native FP4 kernel on default tactics (the FlashInfer autotuner ran but
persisted no cache entries on this build, so there may be headroom); `min_p` is ignored under
speculative decoding in this vLLM (the card's min_p is 0, so no effect); prefix caching is
auto-disabled for this hybrid architecture (no cross-run cache confound); GSM8K first-200 is a
small, easy set and EM differences of a point or two are within sampling noise at temperature
1.0; the base-model column uses a different quantizer (Unsloth Dynamic NVFP4) than UkisAI's
llm-compressor recipe, so it is "Swift-NVFP4 vs base-NVFP4 as people actually run them on Spark",
not a pure fine-tune-only ablation.

## License

Scripts in this repository: MIT (see `LICENSE`). The Swift model weights are **not**
redistributed here; they are gated under UkisAI's Swift Open License 1.0 on Hugging Face.
GSM8K is MIT-licensed by OpenAI and is downloaded, not vendored.
