#!/bin/bash
# ForkLeft evaluation script
# Usage: bash scripts/evaluate.sh --model <checkpoint_path> --datasets <comma-separated>

set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
MODEL="${MODEL:-}"
DATASETS="${DATASETS:-GSM8K,MATH500}"
OUT="${OUT:-eval_output}"

if [ -z "$MODEL" ]; then
    echo "Error: MODEL is required."
    echo "Usage: MODEL=<checkpoint_path> bash scripts/evaluate.sh"
    exit 1
fi

export PYTHONPATH="$REPO:$REPO/src"
mkdir -p "$OUT"

for DS in $(echo "$DATASETS" | tr ',' ' '); do
    echo "[eval] $DS"
    "$PY" -u "$REPO/src/eval_model_zoo.py" \
        --model "$MODEL" --name "forkleft" --family edlm \
        --datasets "$DS" --out "$OUT" \
        --row-shard-index 0 --row-shard-count 1 \
        --row-shard-mode length --prompt-style completion \
        --math-budget 2048 --steps-per-block 32 --batch-size 64
done
echo "[eval] done; results in $OUT"
