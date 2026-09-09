#!/usr/bin/env bash
# Interp pass on the GPT-2 organism models. Models are pulled from Hugging Face ($HF_USER, private repos need HF_TOKEN).
#   bash run_interp.sh smoke                                   # GPT-2, tiny versions of everything
#   bash run_interp.sh                                         # GPT-2, full run
#   TAG=qwen2.5-0.5b BASE=Qwen/Qwen2.5-0.5B bash run_interp.sh   # Qwen
set -euo pipefail
HF_USER=${HF_USER:-WFJKK}
TAG=${TAG:-gpt2}                 # organism tag: gpt2 or qwen2.5-0.5b
BASE=${BASE:-gpt2}               # base repo:    gpt2 or Qwen/Qwen2.5-0.5B
DATA=data/organism
SMOKE=${1:-}
FAM="--tag $TAG --base $BASE"
[ -f $DATA/train.jsonl ] || python gen_cups.py --out $DATA --n-min 16 --n-max 32 --num-train 8000 --test-per-n 60 --extra-test-n 40 --extra-test-count 1000
mkdir -p results runs
if [ "$SMOKE" = "smoke" ]; then
  python interp_boundary.py --user $HF_USER $FAM --data $DATA --out runs/smoke_boundary.json --smoke
  python interp_circuit.py --user $HF_USER $FAM --cond encoded --data $DATA --out runs/smoke_circuit.json --smoke
  echo "SMOKE OK"
  exit 0
fi
python interp_boundary.py --user $HF_USER $FAM --data $DATA --out results/interp_boundary_$TAG.json
python interp_subspace.py --user $HF_USER $FAM --data $DATA --out results/interp_subspace_$TAG.json
for C in encoded plain; do
  python interp_circuit.py --user $HF_USER $FAM --cond $C --data $DATA --out results/interp_circuit_${TAG}_$C.json
  python battery.py --model-path $HF_USER/cups-organism-$TAG-$C --condition $C --data $DATA --out results/horizon_${TAG}_$C.json --conditions horizon
done
echo "ALL DONE"
