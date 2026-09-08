#!/usr/bin/env python3
"""Fine-tune a causal LM on one condition of the cups dataset and evaluate per-n accuracy.

    python train_cups.py tokcheck --model gpt2 --data data/depth
    python train_cups.py train --model gpt2 --condition direct --data data/depth --out runs/gpt2-direct \
        [--limit 200 --epochs 1]        # smoke test
    python train_cups.py eval --model-path runs/gpt2-direct/final --condition direct --data data/depth \
        --out runs/gpt2-direct/results.json [--per-n-limit 10]

Notes
- loss is computed on completion tokens only
- checkpoints every --save-steps; rerun train with --resume to continue after an interruption
- device is auto-detected: cuda (bf16 if supported), mps (fp32), cpu (fp32)
- eval batches examples of equal n, so prompts inside a batch have identical length (no padding)
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from gen_cups import CODE, CONDITIONS, CUPS, chain_metrics, parse_answer, parse_chain


# ---------------------------------------------------------------- helpers

def read_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def describe_device():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        bf = torch.cuda.is_bf16_supported()
        print(f"device: cuda  {p.name}  {p.total_memory / 2**30:.1f} GiB  bf16={'yes' if bf else 'no'}")
        return "cuda", bf
    if torch.backends.mps.is_available():
        print("device: mps (fp32)")
        return "mps", False
    print("device: cpu (fp32)")
    return "cpu", False


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # older transformers
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


def prompt_ids(tok, prompt):
    prefix = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return prefix + tok(prompt, add_special_tokens=False)["input_ids"]


def encode_example(tok, row, cond):
    p = prompt_ids(tok, row["prompt"])
    c = tok(row["completions"][cond], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return p + c, [-100] * len(p) + c


class CupsDataset(Dataset):
    def __init__(self, rows, tok, cond, max_length):
        self.items, self.longest, too_long = [], 0, 0
        for r in rows:
            ids, labels = encode_example(tok, r, cond)
            self.longest = max(self.longest, len(ids))
            if len(ids) > max_length:
                too_long += 1
                continue
            self.items.append({"input_ids": ids, "labels": labels})
        if too_long:
            sys.exit(f"{too_long} examples exceed --max-length {max_length} (longest is {self.longest}); raise it")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


# ---------------------------------------------------------------- tokcheck

def tokcheck(args):
    tok = load_tokenizer(args.model)
    print(f"tokenizer: {args.model}   bos={tok.bos_token!r} eos={tok.eos_token!r} pad={tok.pad_token!r}")
    for s in [" 3", " fine", " swap", "12", "12:", "\n1: fine.", "ball 5"]:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        print(f"  {s!r:14} -> {len(ids)} token(s): {[tok.decode([i]) for i in ids]}")
    single = all(len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1 for w in CODE.values())
    print("  all five slot words single-token with leading space:", single)
    rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    for cond in CONDITIONS:
        longest = max(len(encode_example(tok, r, cond)[0]) for r in rows)
        print(f"  longest sequence, condition {cond:8}: {longest} tokens")


# ---------------------------------------------------------------- train

def train(args):
    set_seed(args.seed)
    device, bf = describe_device()
    tok = load_tokenizer(args.model)
    rows = read_jsonl(os.path.join(args.data, "train.jsonl"), args.limit)
    ds = CupsDataset(rows, tok, args.condition, args.max_length)
    model = load_model(args.model, torch.float32)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    eff_batch = args.batch_size * args.grad_accum
    steps = (len(ds) + eff_batch - 1) // eff_batch * args.epochs
    print(f"model {args.model}: {n_params:.0f}M params | condition {args.condition} | "
          f"{len(ds)} examples, longest {ds.longest} tokens | eff. batch {eff_batch} | "
          f"{args.epochs} epochs = {steps} optimizer steps")

    use_bf16 = device == "cuda" and bf and args.precision in ("auto", "bf16")
    use_fp16 = device == "cuda" and args.precision == "fp16"
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=True,  # plain log lines instead of progress bars (nohup-friendly)
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id))
    last = get_last_checkpoint(args.out) if (args.resume and os.path.isdir(args.out)) else None
    if last:
        print(f"resuming from {last}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print(f"saved {final}   ({(time.time() - t0) / 60:.1f} min)")


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(args):
    device, bf = describe_device()
    dtype = torch.bfloat16 if (device == "cuda" and bf) else torch.float32
    tok = load_tokenizer(args.model_path)
    model = load_model(args.model_path, dtype).to(device).eval()
    test_file = args.test_file or os.path.join(args.data, "test.jsonl")
    rows = read_jsonl(test_file)
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)
    if args.per_n_limit:
        by_n = {n: v[: args.per_n_limit] for n, v in by_n.items()}
    cond = args.condition
    records, table = [], {}
    t0 = time.time()
    for n in sorted(by_n):
        group = by_n[n]
        max_new = 6 if cond == "direct" else 8 + 8 * n
        preds = []
        for i in range(0, len(group), args.batch_size):
            batch = group[i : i + args.batch_size]
            ids = [prompt_ids(tok, r["prompt"]) for r in batch]
            L = max(len(x) for x in ids)
            inp = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for j, x in enumerate(ids):  # left padding
                inp[j, L - len(x):] = torch.tensor(x)
                mask[j, L - len(x):] = 1
            out = model.generate(
                input_ids=inp.to(device), attention_mask=mask.to(device),
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
            texts = tok.batch_decode(out[:, L:], skip_special_tokens=True)
            for r, text in zip(batch, texts):
                pred = parse_answer(text)
                rec = {"id": r["id"], "n": n, "answer": r["answer"], "start": r["start"],
                       "pred": pred, "correct": pred == r["answer"], "text": text}
                if cond in ("plain", "encoded"):
                    chain = parse_chain(text, cond)
                    rec["chain_exact"], rec["link_acc"] = chain_metrics(chain, r["trajectory"])
                preds.append(rec)
        records += preds
        acc = sum(p["correct"] for p in preds) / len(preds)
        start_heur = sum(p["answer"] == p["start"] for p in preds) / len(preds)
        table[n] = {"acc": acc, "start_heur": start_heur, "count": len(preds)}
        if cond in ("plain", "encoded"):
            table[n]["chain_exact"] = sum(p["chain_exact"] for p in preds) / len(preds)
            table[n]["link_acc"] = sum(p["link_acc"] for p in preds) / len(preds)
        extra = f"  chain {table[n]['chain_exact']:.3f}  link {table[n]['link_acc']:.3f}" if "chain_exact" in table[n] else ""
        print(f"n={n:3d}  acc {acc:.3f}  start-heuristic {start_heur:.3f}  (N={len(preds)}){extra}", flush=True)

    overall = sum(p["correct"] for p in records) / len(records)
    print(f"overall acc {overall:.3f} on {len(records)} examples  ({(time.time() - t0) / 60:.1f} min)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_path": args.model_path, "condition": cond, "test_file": test_file,
                   "overall": overall, "per_n": table, "records": records}, f, indent=1)
    print(f"wrote {args.out}")


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokcheck")
    t.add_argument("--model", required=True)
    t.add_argument("--data", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--condition", choices=CONDITIONS, required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=float, default=3)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--max-length", type=int, default=512)
    t.add_argument("--limit", type=int, default=0, help="use only the first N training rows (smoke test)")
    t.add_argument("--save-steps", type=int, default=200)
    t.add_argument("--log-steps", type=int, default=20)
    t.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("eval")
    t.add_argument("--model-path", required=True)
    t.add_argument("--condition", choices=CONDITIONS, required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--test-file", default=None, help="defaults to <data>/test.jsonl")
    t.add_argument("--out", required=True)
    t.add_argument("--batch-size", type=int, default=50)
    t.add_argument("--per-n-limit", type=int, default=0, help="evaluate only the first N examples per n")

    args = ap.parse_args()
    {"tokcheck": tokcheck, "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
