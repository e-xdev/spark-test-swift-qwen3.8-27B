### swift — `ukisai/Swift-Qwen3.8-27B-NVFP4` — conc8: 5 runs × 200 requests @ concurrency 8, max_tokens 8192, effort template-default

| Metric | median (min–max) |
|---|---|
| GSM8K exact match, % | 97.0 (95.0–98.0) |
| Truncated at max_tokens (count) | 0 (0–0) |
| Mean completion tokens (reasoning + answer) | 340 (298–386) |
| Median tok/s per request, decode phase | 16.4 (16.3–16.5) |
| Engine aggregate generation tok/s, steady state (/metrics, running >= concurrency) | 118.4 (116.8–121.1) |
| Engine aggregate generation tok/s over the whole run (incl. ramp/tail) | 111.8 (111.0–117.3) |
| Median time-to-final-answer per question, s | 13.7 (13.5–14.1) |
| p90 time-to-final-answer per question, s | 42.8 (34.9–44.0) |
| Wall time for the whole batch, s (all questions of the run) | 598 (496–683) |
| Median TTFT, s | 0.60 (0.60–0.60) |
| MTP draft-token acceptance, % (engine counters, token-weighted) | 66.0 (65.1–68.4) |
| MTP mean acceptance length | 2.98 (2.95–3.05) |
| Request errors (count) | 0 (0–0) |
| MTP per-position acceptance (mean) | 0.802, 0.653, 0.534 |

### swift — `ukisai/Swift-Qwen3.8-27B-NVFP4` — single: 5 runs × 20 requests @ concurrency 1, max_tokens 8192, effort template-default

| Metric | median (min–max) |
|---|---|
| GSM8K exact match, % | 90.0 (85.0–95.0) |
| Truncated at max_tokens (count) | 0 (0–0) |
| Mean completion tokens (reasoning + answer) | 316 (275–437) |
| Median tok/s per request, decode phase | 18.3 (17.8–18.8) |
| Engine aggregate generation tok/s, steady state (/metrics, running >= concurrency) | 16.9 (16.2–18.0) |
| Engine aggregate generation tok/s over the whole run (incl. ramp/tail) | 16.3 (16.1–17.1) |
| Median time-to-final-answer per question, s | 12.9 (10.2–13.2) |
| p90 time-to-final-answer per question, s | 33.1 (27.4–63.8) |
| Wall time for the whole batch, s (all questions of the run) | 376 (311–531) |
| Median TTFT, s | 0.37 (0.37–0.37) |
| MTP draft-token acceptance, % (engine counters, token-weighted) | 64.3 (62.0–70.3) |
| MTP mean acceptance length | 2.93 (2.86–3.11) |
| Request errors (count) | 0 (0–0) |
| MTP per-position acceptance (mean) | 0.795, 0.644, 0.524 |
