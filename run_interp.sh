#!/usr/bin/env bash
# Interp pass on the GPT-2 organism models. Models are pulled from Hugging Face ($HF_USER, private repos need HF_TOKEN).
#   bash run_interp.sh smoke      # tiny versions of everything, a couple of minutes
#   bash run_interp.sh            # full run
set -euo pipefail
HF_USER=${HF_USER:-WFJKK}
DATA=data/organism
SMOKE=${1:-}
[ -f $DATA/train.jsonl ] || python gen_cups.py --out $DATA --n-min 16 --n-max 32 --num-train 8000 --test-per-n 60 --extra-test-n 40 --extra-test-count 1000
mkdir -p results runs
if [ "$SMOKE" = "smoke" ]; then
  python interp_boundary.py --user $HF_USER --data $DATA --out runs/smoke_boundary.json --smoke
  python interp_circuit.py --user $HF_USER --cond encoded --data $DATA --out runs/smoke_circuit.json --smoke
  echo "SMOKE OK"
  exit 0
fi
python interp_boundary.py --user $HF_USER --data $DATA --out results/interp_boundary.json
for C in encoded plain; do
  python interp_circuit.py --user $HF_USER --cond $C --data $DATA --out results/interp_circuit_$C.json
  python battery.py --model-path $HF_USER/cups-organism-gpt2-$C --condition $C --data $DATA --out results/horizon_gpt2_$C.json --conditions horizon
done
echo "ALL DONE"
