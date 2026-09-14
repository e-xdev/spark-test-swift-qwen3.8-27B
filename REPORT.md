# Swift-Qwen3.8-27B-NVFP4 on NVIDIA DGX Spark (GB10 / SM121) — results

*Measured 13–14 September 2026 on one DGX Spark. Companion to the model card of
[`ukisai/Swift-Qwen3.8-27B-NVFP4`](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4),
whose numbers were taken on an RTX PRO 6000 Blackwell (SM120). Scripts, recipes, raw outputs and
engine-counter snapshots for every run are in this repository.*

## Setup

| | |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10 Grace-Blackwell, 128 GB unified memory (121.7 GiB visible), 273 GB/s, compute capability SM121 — **single node** |
| Serving | vLLM `0.25.1.dev24+g96bb89286.d20260710` (eugr/spark-vllm-docker image), FlashInfer 0.6.15 with sm_121a kernels, FlashAttention 2 |
| NVFP4 kernel | `FlashInferCutlassNvFp4LinearKernel` — the **native FP4 path**. The Marlin env flags from older GB10 guides are not recognised by this vLLM (`Unknown vLLM environment variable detected: VLLM_NVFP4_GEMM_BACKEND`) and are inert |
| Launch | `vllm serve … --max-model-len 262144 --gpu-memory-utilization 0.6 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --speculative-config.method mtp --speculative-config.num_speculative_tokens 3` (see `recipes/`) |
| Protocol | as on the card: GSM8K test, first 200 questions, exact match; temperature 1.0 · top_p 0.95 · top_k 20 · min_p 0; 8 concurrent requests; 8 192-token output cap; template-default reasoning effort; MTP with 3 draft tokens |
| Runs | 5 measured runs per configuration, each preceded by 8 unmeasured warm-up requests; cells are **median (min–max)** over the 5 runs |
| Speed source | engine counters from vLLM `/metrics`, sampled every 10 s: aggregate = Δ`generation_tokens_total`; steady state = windows with `num_requests_running ≥ concurrency`; MTP = Δaccepted / Δdrafted. Per-request tok/s = client-side decode phase (first → last streamed token, TTFT excluded) |
| Grading | extraction rule v3 (README) applied identically to both models by `scripts/regrade.py`; residual misses reviewed by hand (§3) |

## 1. Swift-NVFP4: DGX Spark next to the card's SM120 column

| Metric | UkisAI card — RTX PRO 6000 (SM120, vLLM 0.29.0) | **DGX Spark GB10 (SM121), this repo** |
|---|---|---|
| GSM8K first-200 exact match, % | 96.5 | **97.0 (95.0–98.0)** |
| Truncated at 8 192 tokens | 0 | 0 (0–0) |
| Mean completion tokens (reasoning + answer) | 391 | 340 (298–386) |
| Median tokens/s per request, 8 concurrent | 85 | **16.4 (16.3–16.5)** |
| Engine aggregate generation tok/s, 8 concurrent, steady state | — | 118.4 (116.8–121.1) |
| Median time-to-final-answer per question, 8 concurrent | — | 13.7 s (13.5–14.1) |
| Median TTFT, 8 concurrent | — | 0.60 s |
| MTP draft-token acceptance | 61 % | **66.0 % (65.1–68.4)** |
| MTP mean acceptance length (max 4) | — | 2.98 (2.95–3.05); per position 0.80 / 0.65 / 0.53 |
| Weights on disk | 28.6 GB | 28.6 GB (same files) |
| Weights in GPU memory (vLLM) | ~29 GB | 26.9 GiB (`Model loading took 26.93 GiB`) |
| KV cache available at util 0.6 | — | 39.2 GiB = 581 K tokens |

Single-stream row (concurrency 1, 20 questions × 5 runs; not on the card): median per-request decode
**18.3 tok/s (17.8–18.8)**, engine steady 16.9, MTP acceptance 64.3 %, median time-to-answer 12.9 s.

What this says:

* **It just runs.** No fallback, no environment tricks, coherent output in English and Spanish, 0 truncations, 0 request errors over 2 000 GSM8K generations. The card's Requirements section can list GB10/SM121 with a current vLLM.
* **Accuracy matches the card**: 97.0 % exact match on the same 200 questions vs 96.5 % on SM120, once answers are extracted robustly (§3). The FP4 path on GB10 costs nothing measurable in accuracy.
* **MTP works at least as well as on SM120**: 66 % acceptance vs 61 %. An earlier pass on trivial prompts had shown ~44 %; that was the prompt set, not the hardware.
* **Speed is bandwidth-bound.** 16.4 vs 85 tok/s per request is a 5.2× gap on a device with ~6.6× less memory bandwidth than a PRO 6000. Concurrency is nearly free: per-request speed drops only from 18.3 (single) to 16.4 (8 concurrent) while aggregate throughput rises 7× to 118 tok/s — on Spark, batch.

## 2. Swift vs the base model on the same Spark

Same box, same vLLM, same protocol, same grading. Base = `unsloth/Qwen3.8-27B-NVFP4` (Unsloth Dynamic
quant). This is *not* a pure fine-tune ablation: the two quantization recipes differ, and the
difference matters on a bandwidth-bound device (see notes below the table).

| Metric (8 concurrent, GSM8K-200 × 5) | **Swift-NVFP4 (UkisAI)** | Qwen3.8-27B-NVFP4 (unsloth) |
|---|---|---|
| GSM8K exact match, % | **97.0** (95.0–98.0) | 96.0 (95.0–97.5) |
| Paired outcome (both right / only Swift / only base / neither), 1 000 paired instances | 948 / 16 / 12 / 24 | |
| Mean completion tokens | **340** (298–386) | 427 (385–471) |
| Median tok/s per request, decode phase | 16.4 | **20.0** |
| Engine aggregate tok/s, steady state | 118.4 | **143.8** |
| Median time-to-final-answer per question | 13.7 s | **12.0 s** |
| p90 time-to-final-answer per question | 42.8 s | 41.8 s |
| Wall time for the 200-question batch | **599 s** (496–683) | 699 s (543–721) |
| Median TTFT | 0.60 s | 0.49 s |
| MTP draft-token acceptance | **66.0 %** | 62.2 % |
| MTP mean acceptance length | 2.98 | 2.87 |
| Weights in GPU memory | 26.9 GiB | 22.1 GiB |
| KV cache at util 0.6 | 581 K tokens (39.2 GiB) | 1.29 M tokens (45.0 GiB) — half the bytes per token |

Single-stream: Swift 18.3 tok/s / 316 tokens / 12.9 s median per question; base 24.5 tok/s / 471 tokens / 10.9 s median.

Reading the two speed rows together:

* The unsloth checkpoint is **~22 % faster per token** (20.0 vs 16.4). Its startup log shows FP8
  (`CompressedTensorsW8A8Fp8`) for layers UkisAI keeps in BF16, it is 4.8 GiB smaller in memory, and its
  KV cache uses half the bytes per token — on a 273 GB/s bus every one of those is decode speed. The
  quantization recipe, not the fine-tune, explains this row.
* Swift writes **~20 % fewer tokens** on GSM8K (340 vs 427; the card's 58 % thinking-token reduction is
  measured on harder tasks where reasoning is long).
* Net effect: the **batch finishes ~14 % sooner with Swift** (599 vs 699 s wall for 200 questions), while
  the **median single question is answered slightly sooner by base** (12.0 vs 13.7 s). Swift's saving lands
  in the long-reasoning tail, which is what dominates throughput; on quick questions the per-token speed
  of the FP8-heavier quant wins. An NVFP4 build of Swift with the same FP8/KV recipe would presumably
  keep the token saving and gain the per-token speed.
* MTP acceptance is higher on Swift (66 vs 62 %): the fine-tune did not hurt the draft head.
* Accuracy is a wash: 97.0 vs 96.0 %, and of the 1 000 paired question instances 948 were solved by both, 16 only by Swift, 12 only by base. Swift's shorter reasoning does not cost correctness on this set.

## 3. Grading, and what the residual misses are

Exact match is only as good as the answer extractor. The first grading pass ("last number in the
final answer") scored Swift at 93.0 % — and an audit of the misses showed most were the grader,
not the model: trailing explanations (`**7 dozen eggs** in 4 weeks` → 4), LaTeX thousands
separators (`$70{,}000` → 0), equations in bold (`**$500 + $800 + $130 = $1,430**` → 500). The
rule was revised twice (v2, v3; documented in the README and in every run summary) and applied
retroactively to the *saved* outputs of both models with `scripts/regrade.py`; the original live
scores are kept as `em_pct_live`. Nothing was re-run.

After v3, the misses that remain (Swift 36 of 1 000 instances; base 40) fall into:

| Category | Swift | Base | Examples |
|---|---|---|---|
| Real model errors | 15 | 17 | q119 (salary, wrong on every run for both), q85 (inclusive year count), q45, q182 |
| Hedged double answers — the model states the intended answer, then "if you meant X, it would be Y"; the grader takes the last | 8 | 12 | q16 (230 miles, then 170 straight-line), q184 (25 points, then 100 %), q161, q187 |
| Rounding / unit phrasing | 8 | 5 | q93 (36.4 s vs gold 36), q174 ("1 hour 35 minutes" for 95), q7 ("2 hours 40 minutes") |
| Ambiguous question | 4 | 5 | q12 ("12 years to break even, profit in year 13", gold 13) |
| Residual grader limits | 1 | 2 | q13 (bolded check-line after the answer) |

So the two models are equivalent on this set, and both reproduce the card's 96.5 % within noise.
UkisAI's own extractor may differ in the hedged/rounding cases; that is worth ±1 point.

## 4. Caveats

* Single node, one vLLM build, 5 runs; temperature-1.0 sampling means EM differences of a point or two
  are within noise (min–max shown).
* GSM8K first-200 is small and easy; it was chosen to mirror the card, not to rank models.
* The FlashInfer autotuner ran but persisted no cache on this build ("Falling back to default
  tactics"), so the FP4 GEMM is on default tactics — there may be headroom.
* `min_p` is ignored under speculative decoding in this vLLM (card's min_p is 0, so no effect).
  Prefix caching is auto-disabled for this hybrid architecture (no cross-run cache effect).
* The base-model run logs contain repeated PyTorch `c10d` TCPStore "Broken pipe" warnings (log noise on this image; 0 request errors in every run).
* The base-model column is a different quantizer (FP8 layers, smaller KV per token); see Section 2.

## 5. Reproduce

```bash
git clone https://github.com/em-xdev/spark-test-swift-qwen3.8-27B && cd swift-spark-benchmark && scripts/prepare-gsm8k.sh
cp recipes/*.yaml ~/spark-vllm-docker/recipes/
tmux new -d -s cluster -x 300 -y 50 -c ~/spark-vllm-docker './run-recipe.sh swift-qwen38-nvfp4 --solo; exec bash'
LABEL=swift MODEL=ukisai/Swift-Qwen3.8-27B-NVFP4 scripts/eval-swift-quant.sh all
# stop, launch qwen38-27b-nvfp4-base the same way, then:
LABEL=base MODEL=unsloth/Qwen3.8-27B-NVFP4 scripts/eval-swift-quant.sh all
scripts/regrade.py results/swift; scripts/regrade.py results/base
scripts/report.py compare results/swift results/base --with-card
```

Raw provenance per run: every prompt/answer (`runN.jsonl`), `/metrics` snapshots and 10 s series,
the request body actually sent (`request-sample.json`), and the server startup log with the kernel
selection (`startup-lines.txt`).
