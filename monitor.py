#!/usr/bin/env python3
"""Monitors that know the task but not the code, applied to each rung of the code ladder.

    python monitor.py --user WFJKK --tag gpt2 --base gpt2 --data data/organism --out results/monitor_gpt2.json \
        --conds plain encoded multi shift [--paths multi=runs/organism-gpt2-multi/final shift=...]

For each condition (its model run on its own completion text, teacher-forced), at the ':' before every slot
word, with the environment's true trajectory as labels:
  text monitor        lookup table fitted on N training instances: slot word -> state (word-only features),
                      and (word, line index mod 5) -> state; accuracy on held-out instances, unseen key = wrong
  activation monitor  linear probe on the residual stream at the state layer (earliest layer whose in-model
                      probe with all training instances reaches 0.98), fitted on N instances; accuracy held-out
  unsupervised        k-means (k = 5) on those activations after PCA to 20 dims, no labels: adjusted Rand
                      index against the true state, and against the emitted word class
Two probe targets: the true state s_k, and the emitted word class (for shift, (s_k + k) mod 5; otherwise s_k).
On shift the two differ, which tells whether the model represents the state or only the cipher output.
"""
import argparse
import json
import os

import numpy as np
import torch

from gen_cups import CUPS, encode_slot
from interp_boundary import completion_positions, eval_probe, fit_probe, repo_id
from train_cups import describe_device, load_model, load_tokenizer, read_jsonl

NS = (5, 10, 20, 50, 100, 240)


@torch.no_grad()
def collect_full(model, tok, rows, cond, device):
    feats, state, wclass, word, line, inst = None, [], [], [], [], []
    for r in rows:
        ids, preds, slots = completion_positions(tok, r["prompt"], r["completions"][cond])
        hs = model(torch.tensor([ids], device=device), output_hidden_states=True).hidden_states
        if feats is None:
            feats = [[] for _ in hs]
        for k, (p, sl) in enumerate(zip(preds, slots), 1):
            s = r["trajectory"][k - 1]
            for l, h in enumerate(hs):
                feats[l].append(h[0, p].float().cpu().numpy())
            state.append(s)
            wclass.append((s - 1 + k) % CUPS + 1 if cond == "shift" else s)
            word.append(ids[sl])
            line.append(k)
            inst.append(r["id"])
    return ([np.stack(f) for f in feats], np.array(state), np.array(wclass), np.array(word), np.array(line), np.array(inst))


def lookup_monitor(keys_tr, y_tr, keys_te, y_te):
    table = {}
    for key, y in zip(keys_tr, y_tr):
        table.setdefault(key, []).append(y)
    table = {k: max(set(v), key=v.count) for k, v in table.items()}
    pred = np.array([table.get(k, -1) for k in keys_te])
    return float((pred == y_te).mean())


def kmeans(X, k=5, iters=30, seeds=5, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_inertia = None, np.inf
    for _ in range(seeds):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best


def ari(a, b):
    """Adjusted Rand index."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    M = np.array([[np.sum((a == x) & (b == y)) for y in ub] for x in ua])
    comb = lambda n: n * (n - 1) / 2
    sum_ij = comb(M).sum()
    sum_a, sum_b = comb(M.sum(1)).sum(), comb(M.sum(0)).sum()
    n = comb(len(a))
    expected = sum_a * sum_b / n
    denom = 0.5 * (sum_a + sum_b) - expected
    return float((sum_ij - expected) / denom) if denom else 0.0


def pca(X, d=20):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:d].T


def participation_ratio(X, y):
    mu = np.stack([X[y == s].mean(0) for s in range(1, CUPS + 1)])
    ev = np.linalg.eigvalsh(np.cov((mu - mu.mean(0)).T))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if (ev ** 2).sum() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tag", default="gpt2")
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", nargs="+", default=["plain", "encoded", "multi", "shift"])
    ap.add_argument("--paths", nargs="*", default=[], help="cond=local_path overrides for models not on the Hub")
    ap.add_argument("--n-instances", type=int, default=300)
    args = ap.parse_args()
    paths = dict(p.split("=", 1) for p in args.paths)

    device, _ = describe_device()
    tok = load_tokenizer(args.base)
    rows = sorted(read_jsonl(os.path.join(args.data, "test.jsonl")), key=lambda r: r["id"])[: args.n_instances]
    out = {}
    for cond in args.conds:
        path = paths.get(cond, repo_id(args.user, cond, args.tag, args.base))
        model = load_model(path, torch.float32).to(device).eval()
        F, state, wclass, word, line, inst = collect_full(model, tok, rows, cond, device)
        del model
        torch.cuda.empty_cache()
        uniq = np.unique(inst)
        cut = uniq[int(0.8 * len(uniq))]
        te = inst >= cut
        train_inst = uniq[uniq < cut]

        # state layer: earliest layer whose in-model probe (all training instances) reaches 0.98
        tr_all = inst < cut
        layer_acc = []
        for l in range(len(F)):
            pr = fit_probe(F[l][tr_all], state[tr_all], device)
            layer_acc.append(eval_probe(pr, F[l][te], state[te], device))
        L = next((l for l, a in enumerate(layer_acc) if a >= 0.98), int(np.argmax(layer_acc)))
        X = F[L]

        res = {"state_layer": L, "layer_acc": layer_acc, "n_words": int(len(np.unique(word))), "rows": {}}
        for N in NS:
            sel = np.isin(inst, train_inst[:N])
            row = {
                "text_word": lookup_monitor(word[sel], state[sel], word[te], state[te]),
                "text_word_k": lookup_monitor(list(zip(word[sel], line[sel] % CUPS)), state[sel],
                                              list(zip(word[te], line[te] % CUPS)), state[te]),
                "act_state": eval_probe(fit_probe(X[sel], state[sel], device), X[te], state[te], device),
                "act_wordclass": eval_probe(fit_probe(X[sel], wclass[sel], device), X[te], wclass[te], device),
            }
            res["rows"][N] = row
        Z = pca(X[te])
        lab = kmeans(Z)
        res["unsup_ari_state"] = ari(lab, state[te])
        res["unsup_ari_wordclass"] = ari(lab, wclass[te])
        res["text_ari_state"] = ari(word[te], state[te])
        res["pr_state"] = participation_ratio(X[te], state[te])
        res["pr_wordclass"] = participation_ratio(X[te], wclass[te])
        out[cond] = res

        print(f"\n{cond}: state layer {L} (in-model probe {layer_acc[L]:.3f}), {res['n_words']} distinct slot words, "
              f"participation ratio state {res['pr_state']:.2f} / wordclass {res['pr_wordclass']:.2f}")
        print("   N   text(word)  text(word,k)   act(state)  act(wordclass)")
        for N, row in res["rows"].items():
            print(f"{N:4d}   {row['text_word']:9.3f}   {row['text_word_k']:11.3f}   {row['act_state']:9.3f}   {row['act_wordclass']:13.3f}")
        print(f"unsupervised k-means ARI vs state {res['unsup_ari_state']:.3f}, vs word class {res['unsup_ari_wordclass']:.3f}; "
              f"text words as clusters ARI vs state {res['text_ari_state']:.3f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
