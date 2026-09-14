#!/usr/bin/env bash
# run-side-by-side.sh — Swift-NVFP4 (UkisAI) vs base Qwen3.8-27B-NVFP4 (unsloth), SEQUENTIALLY on
# one DGX Spark (--solo). Never run both servers at once: they would share the 273 GB/s
# memory bus and every throughput number would be meaningless.
#
# This script PAUSES before every destructive step (stopping a server, launching one) and
# always shows the --dry-run render first. Copy the two recipes into ~/spark-vllm-docker/recipes/
# before running. Run from the repo root on the Spark:
#     scripts/run-side-by-side.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SVD="${SVD:-$HOME/spark-vllm-docker}"
TMUX_TARGET="${TMUX_TARGET:-cluster}"
LOG_CMD="${LOG_CMD:-tmux capture-pane -t $TMUX_TARGET -p -J -S -}"
SWIFT_MODEL="ukisai/Swift-Qwen3.8-27B-NVFP4"
BASE_MODEL="unsloth/Qwen3.8-27B-NVFP4"
SWIFT_RECIPE="swift-qwen38-nvfp4"
BASE_RECIPE="qwen38-27b-nvfp4-base"
export TMUX_TARGET LOG_CMD

confirm() { local a; read -r -p "$1 [y/N] " a; [[ "$a" == y || "$a" == Y ]] || { echo "aborted by user"; exit 1; }; }

serving_now() { curl -sf localhost:8000/v1/models 2>/dev/null | grep -q "\"$1\""; }

wait_ready() {  # $1 = model id
  local i
  for i in $(seq 1 180); do
    if serving_now "$1"; then echo "ready: $1"; return 0; fi
    sleep 10
  done
  echo "timeout (30 min) waiting for $1 — check: tmux attach -t $TMUX_TARGET"; return 1
}

launch() {  # $1 = recipe name. run-recipe.sh runs `docker exec ... vllm serve` in the FOREGROUND, so it
            # is started inside a fresh detached tmux session: that pane becomes the engine log (and provenance).
  [[ -f "$SVD/recipes/$1.yaml" ]] || { echo "missing $SVD/recipes/$1.yaml — copy it from $ROOT/recipes/"; exit 1; }
  echo "--- dry run of recipe $1:"
  ( cd "$SVD" && ./run-recipe.sh "$1" --solo --dry-run )
  confirm "Launch recipe '$1' for real (solo) inside tmux session '$TMUX_TARGET'?"
  tmux kill-session -t "$TMUX_TARGET" 2>/dev/null || true
  tmux set-option -g history-limit 50000
  tmux new -d -s "$TMUX_TARGET" -c "$SVD" "./run-recipe.sh $1 --solo; echo '[launcher exited]'; exec bash"
  echo "launched; follow with: tmux attach -t $TMUX_TARGET   (Ctrl-b d to detach; never Ctrl-C in that pane)"
}

stop_server() {
  confirm "STOP the running vLLM server now (docker rm -f vllm_node)?"
  docker rm -f vllm_node >/dev/null 2>&1 || true
  sleep 5
}

capture_startup() {  # $1 = results dir; keep the kernel-path/launch lines as provenance (pane history is 50000 lines)
  mkdir -p "$1"
  sleep 15
  eval "$LOG_CMD" | grep -iE 'marlin|nvfp4|LinearKernel|FlashInferCutlass|non-default args|specul|Unknown vLLM|weights take|KV cache size|autotune' | tail -n 60 > "$1/startup-lines.txt" || true
  echo "--- startup lines saved to $1/startup-lines.txt:"; cat "$1/startup-lines.txt"
}

cd "$ROOT"
[[ -s data/gsm8k_test.jsonl ]] || scripts/prepare-gsm8k.sh
[[ -z "${TMUX:-}" ]] || { echo "run this from a plain shell, not inside tmux (it creates tmux sessions itself)"; exit 1; }

echo "########## 1/2  Swift-NVFP4 (UkisAI) ##########"
if serving_now "$SWIFT_MODEL"; then
  echo "Swift is already serving on :8000 — reusing it."
else
  if curl -sf localhost:8000/v1/models >/dev/null 2>&1; then echo "another model is serving on :8000"; stop_server; fi
  launch "$SWIFT_RECIPE"; wait_ready "$SWIFT_MODEL"; capture_startup "results/swift"
fi
LABEL=swift MODEL="$SWIFT_MODEL" scripts/eval-swift-quant.sh all

echo "########## 2/2  base Qwen3.8-27B-NVFP4 (unsloth) ##########"
if ! serving_now "$BASE_MODEL"; then
  stop_server
  launch "$BASE_RECIPE"; wait_ready "$BASE_MODEL"; capture_startup "results/base"
fi
LABEL=base MODEL="$BASE_MODEL" scripts/eval-swift-quant.sh all

echo "########## compare ##########"
python3 scripts/report.py compare results/swift results/base \
    --labels "Swift-NVFP4 (UkisAI) — DGX Spark" "Qwen3.8-27B-NVFP4 (unsloth) — DGX Spark" --with-card --out results/compare.md
echo "done: results/compare.md  (base model is left running; stop with: docker rm -f vllm_node)"
