#!/usr/bin/env bash
# Stealth ladder on GPT-2: train the multi (3 words per state) and shift (line-dependent) codes, upload them,
# run the task-knowing monitors on plain/encoded/multi/shift, the battery on the new models, and the pairwise
# comparison with the new codes included.
#   nohup bash run_ladder.sh > runs/ladder.log 2>&1 &      (needs HF_TOKEN with write access)
set -euo pipefail
HF_USER=${HF_USER:-WFJKK}
DATA=data/organism
mkdir -p runs results
rm -rf $DATA && python gen_cups.py --out $DATA --n-min 16 --n-max 32 --num-train 8000 --test-per-n 60 --extra-test-n 40 --extra-test-count 1000
python train_cups.py tokcheck --model gpt2 --data $DATA 2>&1 | grep "slot words"
for COND in multi shift; do
  OUT=runs/organism-gpt2-$COND
  if [ ! -d $OUT/final ]; then
    python train_cups.py train --model gpt2 --condition $COND --data $DATA --out $OUT --epochs ${EPOCHS:-3} --batch-size 16 --lr 5e-5 --max-length 512 --resume
    rm -rf $OUT/checkpoint-*
  fi
  python train_cups.py eval --model-path $OUT/final --condition $COND --data $DATA --out results/organism_gpt2_$COND.json
  python train_cups.py eval --model-path $OUT/final --condition $COND --data $DATA --test-file $DATA/test_long.jsonl --out results/organism_gpt2_${COND}_n40.json
  python battery.py --model-path $OUT/final --condition $COND --data $DATA --out results/battery_gpt2_$COND.json
  echo "DONE $COND"
done
python - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN")); user = api.whoami()["name"]
for tag in ("multi", "shift"):
    repo = f"{user}/cups-organism-gpt2-{tag}"
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=f"runs/organism-gpt2-{tag}/final", repo_id=repo, commit_message="final weights")
    print("uploaded", repo, flush=True)
PYEOF
python monitor.py --user $HF_USER --data $DATA --out results/monitor_gpt2.json --conds plain encoded multi shift \
  --paths multi=runs/organism-gpt2-multi/final shift=runs/organism-gpt2-shift/final
python interp_pairs.py --data $DATA --out results/interp_pairs_ladder.json \
  plain=$HF_USER/cups-organism-gpt2-plain:plain enc=$HF_USER/cups-organism-gpt2-encoded:encoded \
  encB=$HF_USER/cups-organism-gpt2-encoded_b-s0:encoded_b multi=runs/organism-gpt2-multi/final:multi \
  shift=runs/organism-gpt2-shift/final:shift --layers 8,9,10,11
echo "ALL DONE"
