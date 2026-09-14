#!/usr/bin/env bash
# run-mock-test.sh — exercise the whole kit against a mock vLLM (no GPU, no network). ~1 minute.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
W=/tmp/swift-mock-test; rm -rf "$W"; mkdir -p "$W"
python3 tests/mock_vllm.py --port 8123 --model ukisai/Swift-Qwen3.8-27B-NVFP4 --accuracy 0.85 > "$W/a.log" 2>&1 & A=$!
python3 tests/mock_vllm.py --port 8124 --model unsloth/Qwen3.8-27B-NVFP4 --accuracy 0.60 --itl 0.05 > "$W/b.log" 2>&1 & B=$!
trap "kill $A $B 2>/dev/null || true" EXIT
sleep 1; python3 tests/mock_log.py reset
export LOG_CMD="python3 tests/mock_log.py" DATA="$ROOT/tests/mini_gsm8k.jsonl" RUNS=3 WARMUP=2 N_CONC=8 CONC=4 N_SINGLE=4 SETTLE=0.2 METRICS_INTERVAL=0.2
BASE_URL=http://localhost:8123/v1 OUT="$W/results/swift" LABEL=swift MODEL=ukisai/Swift-Qwen3.8-27B-NVFP4 scripts/eval-swift-quant.sh all 2>&1 | grep -E "^   run|smoke:|preflight OK"
BASE_URL=http://localhost:8124/v1 OUT="$W/results/base"  LABEL=base  MODEL=unsloth/Qwen3.8-27B-NVFP4      scripts/eval-swift-quant.sh all 2>&1 | grep -E "^   run|smoke:|preflight OK"
python3 scripts/report.py compare "$W/results/swift" "$W/results/base" --with-card --out "$W/results/compare.md" >/dev/null
echo; echo "MOCK TEST OK — see $W/results/compare.md"; head -12 "$W/results/compare.md"
