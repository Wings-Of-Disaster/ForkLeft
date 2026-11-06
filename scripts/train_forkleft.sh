#!/bin/bash
# ForkLeft: Entropy-First Rollouts for Prefix-Aligned AR-to-Diffusion Distillation
#
# Usage:
#   bash scripts/train_forkleft.sh --student <path> --teacher <path> --data <path> --out <path>
#
# Required arguments:
#   --student: path to the DLM student checkpoint (e.g., Efficient-DLM-4B)
#   --teacher: path to the ARM teacher model (e.g., Qwen3-14B or Qwen2.5-Math-7B)
#   --data: path to training data in JSON format
#   --out: output directory for checkpoints and logs
#
# Optional:
#   --teacher2: path to a second teacher for MoE routing (non-math domains)
#   --steps: number of training steps (default: 600)
#   --lr: learning rate (default: 3e-6)
#   --rank-weight: L_rank loss weight (default: 0.1)

set -eu

# Parse arguments
STUDENT="${STUDENT:-}"
TEACHER="${TEACHER:-}"
TEACHER2="${TEACHER2:-}"
DATA="${DATA:-}"
OUT="${OUT:-forkleft_output}"
STEPS="${STEPS:-600}"
SAVE="${SAVE:-50}"
LR="${LR:-3e-6}"
RANKS="${RANKS:-8}"
MASTER_PORT="${MASTER_PORT:-29600}"

if [ -z "$STUDENT" ] || [ -z "$TEACHER" ] || [ -z "$DATA" ]; then
    echo "Error: --student, --teacher, and --data are required."
    echo "Usage: STUDENT=<path> TEACHER=<path> DATA=<path> bash scripts/train_forkleft.sh"
    exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"

export PYTHONPATH="$REPO:$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

mkdir -p "$OUT/logs" "$OUT/run" "$OUT/verdicts"

echo "[forkleft] student=$STUDENT"
echo "[forkleft] teacher=$TEACHER"
echo "[forkleft] data=$DATA"
echo "[forkleft] steps=$STEPS lr=$LR ranks=$RANKS"

TEACHER2_ARG=""
if [ -n "$TEACHER2" ]; then
    TEACHER2_ARG="--teacher2 $TEACHER2 --teacher2-device cuda:0"
    echo "[forkleft] teacher2=$TEACHER2 (MoE routing for non-math domains)"
fi

for r in $(seq 0 $((RANKS - 1))); do
    RANK="$r" CUDA_VISIBLE_DEVICES="$r" setsid nohup "$PY" -u \
        "$REPO/src/train_left_opd.py" \
        --ddp --student "$STUDENT" \
        --teacher "$TEACHER" --teacher-device cuda:0 \
        $TEACHER2_ARG \
        --data "$DATA" --out "$OUT/run" --verdicts "$OUT/verdicts" \
        --gold-min-tokens 32 --train-steps "$STEPS" --batch-rows 2 \
        --gen-length 512 --teacher-gen-length 2048 --max-prompt 512 \
        --sample-ratio 1.0 --max-positions 12 --steps-per-block 32 \
        --position-window 32 --dagger-mode off --divergence-thresh 0.10 \
        --backward-chunk 1 --lr "$LR" --grad-clip 1.0 \
        --log-every 5 --save-every "$SAVE" --seed 42 \
        --fork-loss full --kl-ce-lambda 1.0 --terminate-at-answer \
        --rank-weight 0.1 \
        --student-device cuda:0 \
        > "$OUT/logs/rank$r.log" 2>&1 < /dev/null &
    echo "  rank $r -> GPU $r"
done
echo "[forkleft] launched $RANKS ranks; logs in $OUT/logs/"
