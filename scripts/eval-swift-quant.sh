#!/usr/bin/env bash
# eval-swift-quant.sh — FIXED evaluation wrapper for Swift-Qwen3.8-27B-NVFP4 (and the base
# model) served by vLLM on a single DGX Spark (GB10 / SM121).
#
# What changed vs the first-run script, and why:
#   1. Speed and MTP stats come from the ENGINE: vLLM's /metrics endpoint is sampled every 10 s
#      during each run (generation tokens, drafts, accepted tokens, running requests). Not `time curl`.
#   2. The tmux pane where vLLM logs is captured as a secondary source and for provenance.
#      (`docker logs vllm_node` is always empty under the eugr launcher: vLLM runs via `docker exec`
#      in the foreground of whatever shell ran run-recipe.sh — start that inside `tmux new -d -s cluster`.)
#   3. Adds the 8-concurrent GSM8K-200 run that mirrors the UkisAI card; keeps a clearly
#      labeled single-stream row.
#   4. Sampling = the card's (temp 1.0 / top_p 0.95 / top_k 20 / min_p 0). The first request
#      body of every run is saved verbatim (request-sample.json) as proof.
#   5. RUNS=5 measured runs, each preceded by WARMUP unmeasured requests; report median+min/max.
#   6. The short prompt suite is kept as a SMOKE TEST (garbage/coherence), not an accuracy %.
#
# Usage (run on the Spark, with the model already serving on :8000):
#   LABEL=swift MODEL=ukisai/Swift-Qwen3.8-27B-NVFP4 scripts/eval-swift-quant.sh [preflight|smoke|single|concurrent|report|all]
#   LABEL=base  MODEL=unsloth/Qwen3.8-27B-NVFP4      scripts/eval-swift-quant.sh all
# Env knobs: BASE_URL TMUX_TARGET LOG_CMD RUNS WARMUP EFFORT N_CONC CONC N_SINGLE MAX_TOKENS SETTLE OUT DATA
#   LOG_CMD='' disables pane capture entirely (engine numbers then come from /metrics only)
#   e.g. LOG_CMD="docker exec vllm_node tmux capture-pane -t cluster -p -J -S -"  if tmux lives in the container
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

LABEL="${LABEL:-swift}"
MODEL="${MODEL:-ukisai/Swift-Qwen3.8-27B-NVFP4}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
TMUX_TARGET="${TMUX_TARGET:-cluster}"
LOG_CMD="${LOG_CMD:-tmux capture-pane -t $TMUX_TARGET -p -J -S -}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-8}"
EFFORT="${EFFORT:-}"          # empty = template default (what the NVFP4 card used). e.g. EFFORT=xhigh
N_CONC="${N_CONC:-200}"
CONC="${CONC:-8}"
N_SINGLE="${N_SINGLE:-20}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
SETTLE="${SETTLE:-12}"
METRICS_INTERVAL="${METRICS_INTERVAL:-10}"   # seconds between /metrics samples during a run        # seconds to wait for the last engine log line before snapshotting the pane
DATA="${DATA:-$ROOT/data/gsm8k_test.jsonl}"
OUT="${OUT:-$ROOT/results/$LABEL}"
PY="${PY:-python3}"
BENCH="$HERE/gsm8k-bench.py"
REPORT="$HERE/report.py"
MODE="${1:-all}"

EFFORT_ARGS=()
if [[ -n "$EFFORT" ]]; then EFFORT_ARGS=(--effort "$EFFORT"); fi
COMMON=(--base-url "$BASE_URL" --model "$MODEL" --label "$LABEL" --data "$DATA"
        --max-tokens "$MAX_TOKENS" --log-cmd "$LOG_CMD" --settle "$SETTLE" --metrics-interval "$METRICS_INTERVAL" ${EFFORT_ARGS[@]+"${EFFORT_ARGS[@]}"})

log() { printf '\n== %s ==\n' "$*"; }

preflight() {
  mkdir -p "$OUT"
  log "preflight  label=$LABEL  model=$MODEL"
  command -v "$PY" >/dev/null || { echo "python3 not found"; exit 1; }
  "$PY" "$BENCH" --selftest >/dev/null || { echo "harness selftest FAILED"; exit 1; }
  [[ -s "$DATA" ]] || { echo "GSM8K not found at $DATA — run scripts/prepare-gsm8k.sh first"; exit 1; }
  curl -sf "$BASE_URL/models" > "$OUT/models.json" || { echo "server not reachable at $BASE_URL"; exit 1; }
  grep -q "\"$MODEL\"" "$OUT/models.json" || { echo "model '$MODEL' not in /v1/models:"; cat "$OUT/models.json"; echo; exit 1; }
  curl -sf "${BASE_URL%/v1}/version" > "$OUT/vllm-version.json" 2>/dev/null || echo "(no /version endpoint)"
  if curl -sf "${BASE_URL%/v1}/metrics" > "$OUT/metrics-at-preflight.txt"; then
    echo "engine /metrics reachable: $(grep -cE '^vllm:(generation_tokens_total|spec_decode_num_accepted_tokens_total)' "$OUT/metrics-at-preflight.txt") of 2 key counters present"
  else
    echo "WARNING: ${BASE_URL%/v1}/metrics not reachable — engine-side numbers will rely on the log pane only"
  fi
  if eval "$LOG_CMD" > "$OUT/tmux-pane-at-preflight.txt" 2>/dev/null; then
    echo "captured $(wc -l < "$OUT/tmux-pane-at-preflight.txt") lines from the log pane ($LOG_CMD)"
    if ! grep -q 'Avg generation throughput' "$OUT/tmux-pane-at-preflight.txt"; then
      echo "WARNING: the pane holds no engine throughput lines — is vLLM really logging into '$TMUX_TARGET'?"
    fi
  else
    echo "WARNING: cannot capture a log pane with: $LOG_CMD  (fine if /metrics is reachable)"
    : > "$OUT/tmux-pane-at-preflight.txt"
  fi
  # Provenance: kernel selection + launch args, only present if the pane still holds the startup lines.
  grep -iE 'marlin|nvfp4|LinearKernel|FlashInferCutlass|NVFP4_GEMM|ATOMIC_ADD' "$OUT/tmux-pane-at-preflight.txt" | tail -n 40 > "$OUT/kernel-path.txt" || true
  grep -iE 'non-default args|speculative|num_speculative' "$OUT/tmux-pane-at-preflight.txt" | tail -n 10 > "$OUT/launch-args.txt" || true
  if [[ -s "$OUT/kernel-path.txt" ]]; then
    echo "-- kernel-selection lines (which GEMM path is actually in use):"; cat "$OUT/kernel-path.txt"
  else
    if [[ -s "$OUT/startup-lines.txt" ]]; then
      echo "(kernel-selection lines not in the pane any more; using $OUT/startup-lines.txt captured at launch)"
      grep -E 'LinearKernel|Unknown vLLM|speculative' "$OUT/startup-lines.txt" | head -n 5
    else
      echo "WARNING: no kernel-selection lines in the pane and no startup-lines.txt. Capture them right after a launch:"
      echo "         $LOG_CMD | grep -iE 'marlin|nvfp4|LinearKernel|non-default args' > $OUT/startup-lines.txt"
    fi
  fi
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/nvidia-smi.txt" 2>/dev/null || true
  {
    echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "label: $LABEL"; echo "model: $MODEL"
    echo "base_url: $BASE_URL"; echo "log_cmd: $LOG_CMD"; echo "runs: $RUNS"; echo "warmup: $WARMUP"
    echo "effort: ${EFFORT:-template-default}"; echo "max_tokens: $MAX_TOKENS"
    echo "n_conc: $N_CONC @ $CONC"; echo "n_single: $N_SINGLE @ 1"
    echo "gsm8k_sha256: $(sha256sum "$DATA" | cut -d' ' -f1)"
  } > "$OUT/env.txt"
  echo "preflight OK -> $OUT/env.txt"
}

smoke() {
  log "smoke test (garbage / coherence; single-stream; NOT an accuracy score)"
  "$PY" "$BENCH" "${COMMON[@]}" --prompts "$ROOT/prompts/smoke.jsonl" --n 0 --concurrency 1 --runs 1 --warmup 0 \
      --no-grade --out "$OUT/smoke"
  "$PY" - "$OUT/smoke/run1.jsonl" <<'EOF'
import json, re, sys
n = g = e = 0
for ln in open(sys.argv[1]):
    r = json.loads(ln); n += 1
    c = r.get("content") or ""
    bad = bool(re.search(r"!{5,}", c)) or not c.strip()
    g += bad; e += 1 if r.get("error") else 0
    print(f"[q{r['qid']}] {'GARBAGE' if bad else 'ok':7s} ctok={r.get('completion_tokens')} | {c.strip()[:150].replace(chr(10),' ')}")
print(f"smoke: {n} prompts, garbage={g}, errors={e}")
EOF
}

single() {
  log "single-stream row: $N_SINGLE questions x $RUNS runs @ concurrency 1 (labeled separately from the card's 8-concurrent figure)"
  "$PY" "$BENCH" "${COMMON[@]}" --n "$N_SINGLE" --concurrency 1 --runs "$RUNS" --warmup "$WARMUP" --out "$OUT/single"
}

concurrent() {
  log "card protocol: GSM8K first $N_CONC x $RUNS runs @ concurrency $CONC, max_tokens $MAX_TOKENS"
  "$PY" "$BENCH" "${COMMON[@]}" --n "$N_CONC" --concurrency "$CONC" --runs "$RUNS" --warmup "$WARMUP" --out "$OUT/conc$CONC"
}

report() {
  log "report"
  "$PY" "$REPORT" summarize "$OUT"
}

case "$MODE" in
  preflight)  preflight ;;
  smoke)      preflight; smoke ;;
  single)     preflight; single; report ;;
  concurrent) preflight; concurrent; report ;;
  report)     report ;;
  all)        preflight; smoke; single; concurrent; report ;;
  *) echo "usage: $0 [preflight|smoke|single|concurrent|report|all]"; exit 2 ;;
esac
