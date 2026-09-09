#!/usr/bin/env python3
"""Boundary-relabelling tests on the plain and encoded organism models (GPT-2 by default; --tag/--base for Qwen).

    python interp_boundary.py --user WFJKK --data data/organism --out results/interp_boundary.json \
        [--n-instances 500] [--per-n-eval 20] [--smoke]

1. probe transfer: linear probes for the state after swap k (and for the previous state), read at the token
   before each slot word (where the slot word is predicted), per layer, trained on one model and tested on
   base / plain / encoded. "strict" applies the probe unchanged; "recentred" re-estimates only the feature
   mean and scale on the target model.
2. weight diff: per-block norms of (fine-tune minus base) and the cosine between the plain and encoded deltas;
   the ten slot-word rows of the (tied) embedding matrix are reported separately.
3. row transplant (state-aligned): the plain model with its number-word rows replaced by the encoded
   model's code-word rows for the same states, run on the plain task, and the reverse; controls put the
   base model's rows back in.

Hidden state l is the residual stream entering block l (l = 0 is the embedding output); l = 12 is after ln_f.
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from gen_cups import CODE, PLAIN
from train_cups import describe_device, load_model, load_tokenizer, prompt_ids, read_jsonl

CONDS = ("base", "plain", "encoded")


def repo_id(user, cond, tag="gpt2", base="gpt2"):
    return base if cond == "base" else f"{user}/cups-organism-{tag}-{cond}"


# ---------------------------------------------------------------- positions

def completion_positions(tok, prompt, completion):
    """Full token ids (bos + prompt + completion) and, per slot line, the positions of the ':' token
    (where the slot word is predicted) and of the slot word token."""
    p_ids = prompt_ids(tok, prompt)
    enc = tok(completion, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    def tok_at(char):
        return next(i for i, (a, b) in enumerate(offs) if a <= char < b)

    preds, slots, char = [], [], 0
    for line in completion.split("\n"):
        if line.startswith("ball"):
            break
        colon = char + line.index(":")
        preds.append(len(p_ids) + tok_at(colon))
        slots.append(len(p_ids) + tok_at(colon + 2))
        char += len(line) + 1
    return p_ids + ids, preds, slots


# ---------------------------------------------------------------- activations

@torch.no_grad()
def collect(model, tok, rows, cond, device):
    """Residual stream at every hidden-state index, at the ':' before each slot word."""
    feats, state, prev, inst = None, [], [], []
    for r in rows:
        ids, preds, _ = completion_positions(tok, r["prompt"], r["completions"][cond])
        hs = model(torch.tensor([ids], device=device), output_hidden_states=True).hidden_states
        if feats is None:
            feats = [[] for _ in hs]
        traj = r["trajectory"]
        for k, p in enumerate(preds):
            for l, h in enumerate(hs):
                feats[l].append(h[0, p].float().cpu().numpy())
            state.append(traj[k])
            prev.append(traj[k - 1] if k > 0 else r["start"])
            inst.append(r["id"])
    return [np.stack(f) for f in feats], np.array(state), np.array(prev), np.array(inst)


# ---------------------------------------------------------------- probes

def fit_probe(X, y, device, l2=1e-3):
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y - 1, device=device)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    W = torch.zeros(X.shape[1], 5, device=device, requires_grad=True)
    b = torch.zeros(5, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xn @ W + b, y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach(), mu, sd


def eval_probe(probe, X, y, device, recentre=False):
    W, b, mu, sd = probe
    X = torch.tensor(X, dtype=torch.float32, device=device)
    if recentre:
        mu, sd = X.mean(0), X.std(0) + 1e-6
    pred = ((X - mu) / sd @ W + b).argmax(1).cpu().numpy()
    return float((pred == (y - 1)).mean())


def probe_transfer(acts, device, train_frac=0.8):
    """acts[cond] = (feats per layer, state, prev, inst). Returns {target: {layer: {src->tgt: acc}}}."""
    out = {}
    for target in ("state", "prev"):
        out[target] = {}
        n_layers = len(acts["plain"][0])
        for l in range(n_layers):
            row = {}
            for src in ("plain", "encoded"):
                feats, y = acts[src][0][l], (acts[src][1] if target == "state" else acts[src][2])
                inst = acts[src][3]
                cut = np.unique(inst)[int(train_frac * len(np.unique(inst)))]
                tr, te = inst < cut, inst >= cut
                probe = fit_probe(feats[tr], y[tr], device)
                row[f"{src}->{src}"] = eval_probe(probe, feats[te], y[te], device)
                for tgt in CONDS:
                    if tgt == src:
                        continue
                    tf, ty = acts[tgt][0][l], (acts[tgt][1] if target == "state" else acts[tgt][2])
                    m = acts[tgt][3] >= cut  # same held-out instances
                    row[f"{src}->{tgt}"] = eval_probe(probe, tf[m], ty[m], device)
                    row[f"{src}->{tgt} recentred"] = eval_probe(probe, tf[m], ty[m], device, recentre=True)
            out[target][l] = row
    return out


# ---------------------------------------------------------------- weight diff

def block_of(name):
    m = re.search(r"\.(?:h|layers)\.(\d+)\.(\w+)", name)
    if m:
        kind = m.group(2)
        return f"L{int(m.group(1)):02d}." + ("attn" if "attn" in kind else "mlp" if kind == "mlp" else "ln")
    if "wte" in name or "embed_tokens" in name:
        return "wte (other rows)"
    if "wpe" in name:
        return "wpe"
    if "lm_head" in name:
        return "lm_head (untied)"
    return "ln_f"


def weight_diff(models, word_ids):
    base, plain, enc = models["base"], models["plain"], models["encoded"]
    acc = {}
    for (n, pb), (_, pp), (_, pe) in zip(base.named_parameters(), plain.named_parameters(), enc.named_parameters()):
        dp, de = (pp.detach() - pb.detach()).float(), (pe.detach() - pb.detach()).float()
        if "wte" in n or "embed_tokens" in n:
            mask = torch.zeros(dp.shape[0], dtype=torch.bool, device=dp.device)
            mask[word_ids] = True
            pieces = [("wte (slot-word rows)", dp[mask], de[mask]), ("wte (other rows)", dp[~mask], de[~mask])]
        else:
            pieces = [(block_of(n), dp, de)]
        for g, a, b in pieces:
            a, b = a.flatten(), b.flatten()
            s = acc.setdefault(g, {"pp": 0.0, "ee": 0.0, "pe": 0.0, "d": 0.0})
            s["pp"] += float(a @ a)
            s["ee"] += float(b @ b)
            s["pe"] += float(a @ b)
            s["d"] += float((a - b) @ (a - b))
    table = {}
    for g, s in sorted(acc.items()):
        table[g] = {"norm_plain": s["pp"] ** 0.5, "norm_encoded": s["ee"] ** 0.5,
                    "cosine": s["pe"] / max((s["pp"] * s["ee"]) ** 0.5, 1e-12),
                    "rel_diff": (s["d"] ** 0.5) / max(s["pp"] ** 0.5, 1e-12)}
    return table


# ---------------------------------------------------------------- transplant

def transplant(recipient, donor, recipient_ids, donor_ids):
    """Copy of `recipient` whose embedding rows recipient_ids[i] are replaced by the donor's rows donor_ids[i]
    (state-aligned: the recipient's row for state s gets the donor's row for state s)."""
    m = copy.deepcopy(recipient)
    with torch.no_grad():
        pairs = [(m.get_input_embeddings().weight, donor.get_input_embeddings().weight)]
        if m.get_output_embeddings().weight.data_ptr() != m.get_input_embeddings().weight.data_ptr():  # untied
            pairs.append((m.get_output_embeddings().weight, donor.get_output_embeddings().weight))
        for r_mat, d_mat in pairs:
            for r_id, d_id in zip(recipient_ids, donor_ids):
                r_mat[r_id] = d_mat[d_id].to(r_mat.dtype)
    return m


def run_eval(path, cond, data, out, per_n):
    cmd = [sys.executable, "train_cups.py", "eval", "--model-path", path, "--condition", cond, "--data", data,
           "--out", out, "--per-n-limit", str(per_n)]
    subprocess.run(cmd, check=True)
    r = json.load(open(out))
    recs = r["records"]
    return {"acc": r["overall"], "chain_exact": float(np.mean([x["chain_exact"] for x in recs])),
            "link_acc": float(np.mean([x["link_acc"] for x in recs]))}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Hugging Face user holding the cups-organism repos")
    ap.add_argument("--tag", default="gpt2", help="organism tag: gpt2 or qwen2.5-0.5b")
    ap.add_argument("--base", default="gpt2", help="base model repo: gpt2 or Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-instances", type=int, default=500)
    ap.add_argument("--per-n-eval", type=int, default=20, help="instances per n for the transplant evals")
    ap.add_argument("--smoke", action="store_true", help="20 instances, 3 per n")
    args = ap.parse_args()
    if args.smoke:
        args.n_instances, args.per_n_eval = 20, 3

    device, _ = describe_device()
    tok = load_tokenizer(args.base)
    tok.padding_side = "left"
    models = {c: load_model(repo_id(args.user, c, args.tag, args.base), torch.float32).to(device).eval() for c in CONDS}
    rows = read_jsonl(os.path.join(args.data, "test.jsonl"))
    rows = sorted(rows, key=lambda r: r["id"])[: args.n_instances]
    word_ids = {w: tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in list(PLAIN.values()) + list(CODE.values())}
    results = {"n_instances": len(rows)}
    t0 = time.time()

    # 1. probes ------------------------------------------------------------------------
    acts = {}
    for c in ("plain", "encoded"):
        acts[c] = collect(models[c], tok, rows, c, device)
    # base model on each text; transfer is tested on the matching text
    acts["base"] = collect(models["base"], tok, rows, "plain", device)
    acts_base_enc = collect(models["base"], tok, rows, "encoded", device)
    probes = probe_transfer({"plain": acts["plain"], "encoded": acts["encoded"], "base": acts["base"]}, device)
    # encoded-trained probe on base run over encoded text
    for target in ("state", "prev"):
        for l in probes[target]:
            feats, y = acts["encoded"][0][l], (acts["encoded"][1] if target == "state" else acts["encoded"][2])
            inst = acts["encoded"][3]
            cut = np.unique(inst)[int(0.8 * len(np.unique(inst)))]
            probe = fit_probe(feats[inst < cut], y[inst < cut], device)
            m = acts_base_enc[3] >= cut
            ty = acts_base_enc[1] if target == "state" else acts_base_enc[2]
            probes[target][l]["encoded->base(enc text)"] = eval_probe(probe, acts_base_enc[0][l][m], ty[m], device)
    results["probes"] = probes
    for target in ("state", "prev"):
        keys = ["plain->plain", "plain->encoded", "plain->encoded recentred", "plain->base", "encoded->encoded",
                "encoded->plain", "encoded->plain recentred", "encoded->base(enc text)"]
        print(f"\nprobe accuracy, target = {target} (chance 0.2)")
        print("layer  " + "  ".join(f"{k:>26}" for k in keys))
        for l, row in probes[target].items():
            print(f"{l:5d}  " + "  ".join(f"{row.get(k, float('nan')):26.3f}" for k in keys))
    print(f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    # 2. weight diff ---------------------------------------------------------------------
    wd = weight_diff(models, torch.tensor(list(word_ids.values()), device=device))
    results["weight_diff"] = wd
    print("\nweight diff vs base (norms; cosine between plain and encoded deltas; |plain-enc|/|plain|)")
    for g, s in wd.items():
        print(f"{g:22} plain {s['norm_plain']:8.3f}  encoded {s['norm_encoded']:8.3f}  cos {s['cosine']:6.3f}  rel_diff {s['rel_diff']:6.3f}")

    # 3. transplant ----------------------------------------------------------------------
    code_ids = [word_ids[w] for w in CODE.values()]
    plain_ids = [word_ids[w] for w in PLAIN.values()]
    os.makedirs("runs", exist_ok=True)
    B, P, E = models["base"], models["plain"], models["encoded"]
    hybrids = {
        "plain, number rows <- encoded code rows":  (transplant(P, E, plain_ids, code_ids), "plain"),
        "encoded, code rows <- plain number rows":  (transplant(E, P, code_ids, plain_ids), "encoded"),
        "plain, number rows <- base number rows":   (transplant(P, B, plain_ids, plain_ids), "plain"),
        "encoded, code rows <- base code rows":     (transplant(E, B, code_ids, code_ids), "encoded"),
        "plain, number rows <- base code rows":     (transplant(P, B, plain_ids, code_ids), "plain"),
        "plain, untouched":                         (P, "plain"),
        "encoded, untouched":                       (E, "encoded"),
    }
    results["transplant"] = {}
    print("\ntransplant (state-aligned rows; free generation on the recipient's own task)")
    for i, (name, (m, cond)) in enumerate(hybrids.items()):
        path = f"runs/hybrid-{i}"
        m.save_pretrained(path)
        tok.save_pretrained(path)
        res = run_eval(path, cond, args.data, path + "/results.json", args.per_n_eval)
        results["transplant"][name] = res
        print(f"{name:42} acc {res['acc']:.3f}  chain_exact {res['chain_exact']:.3f}  link_acc {res['link_acc']:.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}  [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
