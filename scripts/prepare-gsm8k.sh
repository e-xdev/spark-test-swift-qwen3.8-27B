#!/usr/bin/env bash
# prepare-gsm8k.sh — fetch the GSM8K test split (1319 rows) as data/gsm8k_test.jsonl and
# record its sha256 for provenance. The benchmark uses the FIRST 200 rows (card protocol)
# plus rows 200..207 as unmeasured warmup.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data"
OUT="$ROOT/data/gsm8k_test.jsonl"

if [[ -s "$OUT" ]]; then
  echo "already present: $OUT"
else
  # Canonical source (openai/grade-school-math on GitHub). The HF dataset openai/gsm8k 'main'
  # test split is the same 1319 rows in the same order.
  URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
  echo "downloading $URL"
  if ! curl -fsSL "$URL" -o "$OUT.tmp"; then
    echo "curl failed; falling back to HF datasets (pip install datasets --break-system-packages)"
    python3 - "$OUT.tmp" <<'EOF'
import json, sys
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
with open(sys.argv[1], "w") as f:
    for r in ds:
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
EOF
  fi
  mv "$OUT.tmp" "$OUT"
fi

python3 - "$OUT" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert len(rows) == 1319, f"expected 1319 rows, got {len(rows)}"
assert rows[0]["question"].startswith("Janet"), rows[0]["question"][:80]
assert rows[0]["answer"].strip().endswith("#### 18"), rows[0]["answer"][-40:]
print(f"GSM8K test split OK: {len(rows)} rows; q0 = Janet's ducks -> 18")
EOF
sha256sum "$OUT" | tee "$ROOT/data/gsm8k_test.sha256"
