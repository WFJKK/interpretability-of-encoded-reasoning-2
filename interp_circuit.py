#!/usr/bin/env python3
"""Per-link circuit of an organism model, with TransformerLens (GPT-2 by default; --tag/--base for Qwen).

    python interp_circuit.py --user WFJKK --cond encoded --data data/organism \
        --out results/interp_circuit_encoded.json [--n-attn 200 --n-patch 100] [--smoke]

4. attention: for every head, mean attention from the ':' before slot k to (a) the previous slot word,
   (b) the tokens of prompt line k (index, swap, a, b), (c) everything else. Heads are ranked as
   state-carry (a) or swap-read (b).
5. patching: a corrupted run in which slot word k-1 encodes a different state. Metric: logit(word for the
   clean state after swap k) minus logit(word for the corrupted one), at the ':' before slot k. Recovery is
   the fraction of the clean-minus-corrupted gap restored by copying the clean activation into the corrupted
   run, for (a) the residual stream entering each layer at each of the five positions from slot word k-1 to
   the ':' (layer 0 at slot word k-1 = restoring the token identity), (b) each head's output at the ':',
   (c) each MLP's output at the ':'.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from gen_cups import CONDITIONS, CUPS, encode_slot
from interp_boundary import completion_positions, repo_id
from train_cups import describe_device, load_model, load_tokenizer, read_jsonl

SPAN = ["slot k-1", ".", "newline", "index k", ":"]


def prompt_line_positions(tok, prompt):
    """Per swap line of the prompt, the token positions (in bos + prompt) covering that line."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    off = 1 if tok.bos_token_id is not None else 0
    out, char = [], 0
    for line in prompt.split("\n"):
        if "swap" in line:
            a, b = char, char + len(line)
            out.append([off + i for i, (x, y) in enumerate(offs) if x < b and y > a])
        char += len(line) + 1
    return out


def load_tl(user, cond, tok, device, tag="gpt2", base="gpt2"):
    from transformer_lens import HookedTransformer
    hf = load_model(repo_id(user, cond, tag, base), torch.float32)
    model = HookedTransformer.from_pretrained(base, hf_model=hf, tokenizer=tok, device=device)
    model.eval()
    return model


@torch.no_grad()
def attention_map(model, tok, rows, cond):
    L, H = model.cfg.n_layers, model.cfg.n_heads
    acc, count = np.zeros((3, L, H)), 0
    for r in rows:
        ids, preds, slots = completion_positions(tok, r["prompt"], r["completions"][cond])
        plines = prompt_line_positions(tok, r["prompt"])
        toks = torch.tensor([ids], device=model.cfg.device)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("attn.hook_pattern"))
        for k in range(1, len(preds)):
            q, prev, line = preds[k], slots[k - 1], plines[k]
            for l in range(L):
                pat = cache[f"blocks.{l}.attn.hook_pattern"][0, :, q, :].cpu().numpy()  # [H, key]
                a, b = pat[:, prev], pat[:, line].sum(-1)
                acc[0, l] += a
                acc[1, l] += b
                acc[2, l] += 1 - a - b
            count += 1
    return acc / max(count, 1)


def metric(logits, pos, clean_id, corr_id):
    return (logits[0, pos, clean_id] - logits[0, pos, corr_id]).item()


@torch.no_grad()
def patching(model, tok, rows, cond, rng):
    def wid(k, s):  # token id of the slot word for state s on line k
        return tok(" " + encode_slot(cond, k, s), add_special_tokens=False)["input_ids"][0]
    L, H = model.cfg.n_layers, model.cfg.n_heads
    resid, heads, mlps = np.zeros((L, len(SPAN))), np.zeros((L, H)), np.zeros(L)
    gaps, count = [], 0
    for r in rows:
        ids, preds, slots = completion_positions(tok, r["prompt"], r["completions"][cond])
        n = len(preds)
        k = rng.randint(2, n)  # 1-based line index with a previous slot word
        s_prev, s_clean = r["trajectory"][k - 2], r["trajectory"][k - 1]
        s_corr_prev = rng.choice([s for s in range(1, CUPS + 1) if s != s_prev])
        a, b = r["swaps"][k - 1]
        s_corr = b if s_corr_prev == a else a if s_corr_prev == b else s_corr_prev
        q = preds[k - 1]
        s0 = slots[k - 2]
        # position groups: slot word k-1, '.', newline, the index token(s) of line k, ':'
        groups = [[s0], [s0 + 1], [s0 + 2], list(range(s0 + 3, q)), [q]]
        if not groups[3]:
            continue
        clean = torch.tensor([ids], device=model.cfg.device)
        corr = clean.clone()
        corr[0, slots[k - 2]] = wid(k - 1, s_corr_prev)
        w_clean, w_corr = wid(k, s_clean), wid(k, s_corr)
        clean_logits, ccache = model.run_with_cache(clean)
        mc = metric(clean_logits, q, w_clean, w_corr)
        mx = metric(model(corr), q, w_clean, w_corr)
        gaps.append({"id": r["id"], "k": k, "clean": mc, "corrupted": mx})
        if mc - mx <= 0:
            continue

        def rec(logits):
            return (metric(logits, q, w_clean, w_corr) - mx) / (mc - mx)

        for l in range(L):
            name = f"blocks.{l}.hook_resid_pre"
            for j, poss in enumerate(groups):
                def hook(act, hook, poss=poss, name=name):
                    act[:, poss] = ccache[name][:, poss]
                    return act
                resid[l, j] += rec(model.run_with_hooks(corr, fwd_hooks=[(name, hook)]))
            zname = f"blocks.{l}.attn.hook_z"
            for h in range(H):
                def hook(act, hook, h=h, zname=zname):
                    act[:, q, h] = ccache[zname][:, q, h]
                    return act
                heads[l, h] += rec(model.run_with_hooks(corr, fwd_hooks=[(zname, hook)]))
            mname = f"blocks.{l}.hook_mlp_out"

            def hook(act, hook, mname=mname):
                act[:, q] = ccache[mname][:, q]
                return act
            mlps[l] += rec(model.run_with_hooks(corr, fwd_hooks=[(mname, hook)]))
        count += 1
    return resid / max(count, 1), heads / max(count, 1), mlps / max(count, 1), gaps, count


def top(mat, k=8):
    idx = np.argsort(mat.flatten())[::-1][:k]
    return [(f"L{i // mat.shape[1]}H{i % mat.shape[1]}", float(mat.flatten()[i])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tag", default="gpt2")
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--cond", choices=[c for c in CONDITIONS if c not in ("direct", "random")], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-attn", type=int, default=200)
    ap.add_argument("--n-patch", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_attn, args.n_patch = 10, 3

    device, _ = describe_device()
    tok = load_tokenizer(args.base)
    model = load_tl(args.user, args.cond, tok, device, args.tag, args.base)
    rows = sorted(read_jsonl(os.path.join(args.data, "test.jsonl")), key=lambda r: r["id"])
    rng = random.Random(args.seed)
    t0 = time.time()

    att = attention_map(model, tok, rows[: args.n_attn], args.cond)
    print(f"\nattention from ':' before slot k, mean over lines ({args.cond} model)")
    print("state-carry heads (to previous slot word):", ", ".join(f"{h} {v:.2f}" for h, v in top(att[0])))
    print("swap-read heads   (to prompt line k):     ", ", ".join(f"{h} {v:.2f}" for h, v in top(att[1])))
    print(f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    resid, heads, mlps, gaps, count = patching(model, tok, rows[: args.n_patch], args.cond, rng)
    mc = np.mean([g["clean"] for g in gaps])
    mx = np.mean([g["corrupted"] for g in gaps])
    print(f"\npatching on {count} instances; mean metric clean {mc:.2f}, corrupted {mx:.2f}")
    print("residual-stream recovery, rows = layer (resid_pre), columns = position")
    print("layer  " + "  ".join(f"{s:>9}" for s in SPAN))
    for l in range(resid.shape[0]):
        print(f"{l:5d}  " + "  ".join(f"{v:9.2f}" for v in resid[l]))
    print("head-output recovery at ':' :", ", ".join(f"{h} {v:.2f}" for h, v in top(heads)))
    print("MLP-output recovery at ':'  :", ", ".join(f"L{l} {v:.2f}" for l, v in enumerate(mlps)))
    print(f"[{(time.time() - t0) / 60:.1f} min]")

    with open(args.out, "w") as f:
        json.dump({"cond": args.cond, "n_attn": args.n_attn, "n_patch": count,
                   "attention": {"to_prev_slot": att[0].tolist(), "to_prompt_line": att[1].tolist(), "other": att[2].tolist()},
                   "patching": {"resid_pre": resid.tolist(), "span": SPAN, "heads_at_colon": heads.tolist(),
                                "mlp_at_colon": mlps.tolist(), "gaps": gaps}}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
