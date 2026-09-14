## Swift-NVFP4 (UkisAI) vs Qwen3.8-27B-NVFP4 (unsloth) — same DGX Spark (GB10, SM121, native FlashInfer CUTLASS NVFP4 kernel), same vLLM, same protocol

Protocol: GSM8K test first 200 questions, exact match; 5 measured runs (median, min–max); concurrency 8; max_tokens 8192; effort template-default; sampling temp 1.0 / top_p 0.95 / top_k 20 / min_p 0; MTP method=mtp, 3 draft tokens.

| Metric | Swift-NVFP4 (UkisAI) | Qwen3.8-27B-NVFP4 (unsloth) | Swift-NVFP4, UkisAI card (RTX PRO 6000 Blackwell SM120, vLLM 0.29.0) |
|---|---|---|---|
| GSM8K exact match, % | 97.0 (95.0–98.0) | 96.0 (95.0–97.5) | 96.5 |
| Truncated at max_tokens (count) | 0 (0–0) | 0 (0–1) | 0 |
| Mean completion tokens (reasoning + answer) | 340 (298–386) | 427 (385–471) | 391 |
| Median tok/s per request, decode phase | 16.4 (16.3–16.5) | 20.0 (19.9–20.1) | 85 (card: 'median tokens/s per request, 8 concurrent') |
| Engine aggregate generation tok/s, steady state (/metrics, running >= concurrency) | 118.4 (116.8–121.1) | 143.8 (141.9–146.1) | — |
| Engine aggregate generation tok/s over the whole run (incl. ramp/tail) | 111.8 (111.0–117.3) | 120.0 (117.6–138.9) | — |
| Median time-to-final-answer per question, s | 13.7 (13.5–14.1) | 12.0 (11.8–12.2) | — |
| p90 time-to-final-answer per question, s | 42.8 (34.9–44.0) | 41.8 (39.8–44.4) | — |
| Wall time for the whole batch, s (all questions of the run) | 598 (496–683) | 699 (543–721) | — |
| Median TTFT, s | 0.60 (0.60–0.60) | 0.49 (0.49–0.49) | — |
| MTP draft-token acceptance, % (engine counters, token-weighted) | 66.0 (65.1–68.4) | 62.2 (61.8–63.9) | 61 |
| MTP mean acceptance length | 2.98 (2.95–3.05) | 2.87 (2.85–2.92) | — |
| Request errors (count) | 0 (0–0) | 0 (0–0) | — |
| Paired outcome (both right / only Swift-NVFP4 (UkisAI) / only Qwen3.8-27B-NVFP4 (unsloth)) | 948 / 16 / 12 (neither: 24; 1000 paired instances over 5 runs) | ← same | 192 / 4 / 1 (card: BF16 vs NVFP4) |
| MTP per-position acceptance (mean) | [0.802, 0.653, 0.534] | [0.773, 0.614, 0.493] | — |
| **Single-stream** median decode tok/s (concurrency 1, 20 q × 5 runs) | 18.3 (17.8–18.8) | 24.5 (23.8–25.1) | not on card |
| **Single-stream** MTP acceptance, % | 64.3 (62.0–70.3) | 62.9 (60.2–64.4) | — |
| Weights in GPU memory (startup log `Model loading took N GiB`) | see results/<label>/startup-lines.txt | see results/<label>/startup-lines.txt | ~29 GB |

Notes: per-request tok/s is decode-phase (first→last streamed token), so TTFT is excluded; the engine's own generation-token counters (/metrics, sampled every 10 s) give the aggregate over all running requests and are reported separately. MTP figures are token-weighted deltas of the engine's spec-decode counters over each run. Exact-match extraction rule: v3: #### > last \boxed{} > 'answer is/:' > bold on last bolded line (segment with '=' preferred) > last number; inside a segment: number after last '=' > last $-amount > first number; LaTeX {,} separators removed; '-' after a word char is not a minus. Raw per-question outputs and engine-counter snapshots are in each results dir.

Questions never solved in any run — Swift-NVFP4 (UkisAI): [93, 119]; Qwen3.8-27B-NVFP4 (unsloth): [12, 85, 119]
