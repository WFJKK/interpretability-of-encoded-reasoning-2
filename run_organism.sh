#!/usr/bin/env bash
# Organism: matched fine-tunes on n in 16..32, evaluated on test (16..32) and test_long (n=40).
#   nohup bash run_organism.sh gpt2 > runs/organism_gpt2.log 2>&1 &
#   bash run_organism.sh Qwen/Qwen2.5-0.5B "encoded plain"      # subset of conditions
# Reruns resume; finished conditions are only re-evaluated. Checkpoints are deleted once final is saved.
set -euo pipefail
MODEL=${1:?usage: run_organism.sh MODEL [conditions]}
CONDS=${2:-"direct plain encoded random"}
TAG=$(basename "$MODEL" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9.\n' '-')
DATA=data/organism
EPOCHS=${EPOCHS:-3}
case "$MODEL" in
  gpt2*) BS=16; ACC=1; LR=5e-5; MAXLEN=512 ;;
  *)     BS=8;  ACC=2; LR=2e-5; MAXLEN=768 ;;
esac
mkdir -p runs results
[ -f $DATA/train.jsonl ] || python gen_cups.py --out $DATA --n-min 16 --n-max 32 --num-train 8000 --test-per-n 60 --extra-test-n 40 --extra-test-count 1000
for COND in $CONDS; do
  OUT=runs/organism-$TAG-$COND
  if [ ! -d $OUT/final ]; then
    python train_cups.py train --model "$MODEL" --condition $COND --data $DATA --out $OUT \
      --epochs $EPOCHS --batch-size $BS --grad-accum $ACC --lr $LR --max-length $MAXLEN --resume
    rm -rf $OUT/checkpoint-*
  fi
  python train_cups.py eval --model-path $OUT/final --condition $COND --data $DATA --out results/organism_${TAG}_${COND}.json
  python train_cups.py eval --model-path $OUT/final --condition $COND --data $DATA --test-file $DATA/test_long.jsonl --out results/organism_${TAG}_${COND}_n40.json
  echo "DONE $TAG $COND"
done
echo "ALL DONE $TAG"
