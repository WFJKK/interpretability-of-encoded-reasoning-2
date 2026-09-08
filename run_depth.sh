#!/usr/bin/env bash
# Depth measurement: train the direct condition on n in 1..64, evaluate per-n accuracy.
#   nohup bash run_depth.sh gpt2 > runs/depth_gpt2.log 2>&1 &
#   nohup bash run_depth.sh Qwen/Qwen2.5-0.5B > runs/depth_qwen2.5-0.5b.log 2>&1 &
# Re-running the same command resumes from the last checkpoint; a finished run only re-evaluates.
set -euo pipefail
MODEL=${1:?usage: run_depth.sh MODEL}
TAG=$(basename "$MODEL" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9.\n' '-')
DATA=data/depth
OUT=runs/depth-$TAG
EPOCHS=${EPOCHS:-3}
case "$MODEL" in
  gpt2*) BS=16; ACC=1; LR=5e-5; MAXLEN=512 ;;
  *)     BS=8;  ACC=2; LR=2e-5; MAXLEN=768 ;;
esac
mkdir -p runs results
[ -f $DATA/train.jsonl ] || python gen_cups.py --out $DATA --n-min 1 --n-max 64 --num-train 20000 --test-per-n 100
[ -d $OUT/final ] || python train_cups.py train --model "$MODEL" --condition direct --data $DATA --out $OUT \
    --epochs $EPOCHS --batch-size $BS --grad-accum $ACC --lr $LR --max-length $MAXLEN --resume
python train_cups.py eval --model-path $OUT/final --condition direct --data $DATA --out results/depth_$TAG.json
echo "DONE $TAG"
