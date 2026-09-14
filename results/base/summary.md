### base — `unsloth/Qwen3.8-27B-NVFP4` — conc8: 5 runs × 200 requests @ concurrency 8, max_tokens 8192, effort template-default

| Metric | median (min–max) |
|---|---|
| GSM8K exact match, % | 96.0 (95.0–97.5) |
| Truncated at max_tokens (count) | 0 (0–1) |
| Mean completion tokens (reasoning + answer) | 427 (385–471) |
| Median tok/s per request, decode phase | 20.0 (19.9–20.1) |
| Engine aggregate generation tok/s, steady state (/metrics, running >= concurrency) | 143.8 (141.9–146.1) |
| Engine aggregate generation tok/s over the whole run (incl. ramp/tail) | 120.0 (117.6–138.9) |
| Median time-to-final-answer per question, s | 12.0 (11.8–12.2) |
| p90 time-to-final-answer per question, s | 41.8 (39.8–44.4) |
| Wall time for the whole batch, s (all questions of the run) | 699 (543–721) |
| Median TTFT, s | 0.49 (0.49–0.49) |
| MTP draft-token acceptance, % (engine counters, token-weighted) | 62.2 (61.8–63.9) |
| MTP mean acceptance length | 2.87 (2.85–2.92) |
| Request errors (count) | 0 (0–0) |
| MTP per-position acceptance (mean) | 0.773, 0.614, 0.493 |

### base — `unsloth/Qwen3.8-27B-NVFP4` — single: 5 runs × 20 requests @ concurrency 1, max_tokens 8192, effort template-default

| Metric | median (min–max) |
|---|---|
| GSM8K exact match, % | 90.0 (90.0–95.0) |
| Truncated at max_tokens (count) | 0 (0–0) |
| Mean completion tokens (reasoning + answer) | 471 (416–531) |
| Median tok/s per request, decode phase | 24.5 (23.8–25.1) |
| Engine aggregate generation tok/s, steady state (/metrics, running >= concurrency) | 22.5 (21.5–22.7) |
| Engine aggregate generation tok/s over the whole run (incl. ramp/tail) | 21.9 (21.4–22.2) |
| Median time-to-final-answer per question, s | 10.9 (10.0–11.5) |
| p90 time-to-final-answer per question, s | 36.5 (32.9–48.1) |
| Wall time for the whole batch, s (all questions of the run) | 421 (368–483) |
| Median TTFT, s | 0.27 (0.27–0.29) |
| MTP draft-token acceptance, % (engine counters, token-weighted) | 62.9 (60.2–64.4) |
| MTP mean acceptance length | 2.89 (2.81–2.93) |
| Request errors (count) | 0 (0–0) |
| MTP per-position acceptance (mean) | 0.768, 0.61, 0.496 |
