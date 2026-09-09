#!/usr/bin/env bash
# Seed and code controls on GPT-2: plain seed 1, encoded seed 1, and a second code (encoded_b) seed 0,
# then pairwise probe transfer and subspace angles against the original plain and encoded models.
#   nohup bash run_controls.sh > runs/controls.log 2>&1 &     (needs HF_TOKEN with write access)
set -euo pipefail
HF_USER=${HF_USER:-WFJKK}
DATA=data/organism
mkdir -p runs results
# regenerate so the rows carry the encoded_b completion (same seed, identical instances)
rm -rf $DATA && python gen_cups.py --out $DATA --n-min 16 --n-max 32 --num-train 8000 --test-per-n 60 --extra-test-n 40 --extra-test-count 1000
train() {  # condition seed tag
  OUT=runs/organism-gpt2-$3
  if [ ! -d $OUT/final ]; then
    python train_cups.py train --model gpt2 --condition $1 --data $DATA --out $OUT --epochs 3 --batch-size 16 --lr 5e-5 --max-length 512 --seed $2 --resume
    rm -rf $OUT/checkpoint-*
  fi
  python train_cups.py eval --model-path $OUT/final --condition $1 --data $DATA --out results/organism_gpt2_$3.json --per-n-limit 20
  echo "DONE $3"
}
train plain 1 plain-s1
train encoded 1 encoded-s1
train encoded_b 0 encoded_b-s0
python - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN")); user = api.whoami()["name"]
for tag in ("plain-s1", "encoded-s1", "encoded_b-s0"):
    repo = f"{user}/cups-organism-gpt2-{tag}"
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=f"runs/organism-gpt2-{tag}/final", repo_id=repo, commit_message="final weights")
    print("uploaded", repo, flush=True)
PYEOF
python interp_pairs.py --data $DATA --out results/interp_pairs.json \
  plain0=$HF_USER/cups-organism-gpt2-plain:plain plain1=runs/organism-gpt2-plain-s1/final:plain \
  enc0=$HF_USER/cups-organism-gpt2-encoded:encoded enc1=runs/organism-gpt2-encoded-s1/final:encoded \
  encB=runs/organism-gpt2-encoded_b-s0/final:encoded_b base=gpt2:plain
echo "ALL DONE"
