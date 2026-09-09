#!/usr/bin/env python3
"""Do the plain and encoded models share a state subspace? Per layer, at the ':' before each slot word:
the 4-dim subspace spanned by the five class means of the residual stream in each model, the principal
angles between the two subspaces (cosines; 1 = identical direction), the same against the base model on
each text, and a joint probe trained on both models' activations together (per-model mean removed).

    python interp_subspace.py --user WFJKK --data data/organism --out results/interp_subspace.json
"""
import argparse, json, os
import numpy as np, torch
from interp_boundary import CONDS, collect, eval_probe, fit_probe, repo_id
from train_cups import describe_device, load_model, load_tokenizer, read_jsonl


def basis(X, y):
    mu = np.stack([X[y == s].mean(0) for s in range(1, 6)])
    q, _ = np.linalg.qr((mu - mu.mean(0)).T)
    return q[:, :4]


def cosines(qa, qb):
    return np.linalg.svd(qa.T @ qb, compute_uv=False).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--tag", default="gpt2"); ap.add_argument("--base", default="gpt2")
    args = ap.parse_args()
    device, _ = describe_device()
    tok = load_tokenizer(args.base)
    models = {c: load_model(repo_id(args.user, c, args.tag, args.base), torch.float32).to(device).eval() for c in CONDS}
    rows = sorted(read_jsonl(os.path.join(args.data, "test.jsonl")), key=lambda r: r["id"])[: args.n_instances]
    A = {"plain": collect(models["plain"], tok, rows, "plain", device),
         "encoded": collect(models["encoded"], tok, rows, "encoded", device),
         "base_plain": collect(models["base"], tok, rows, "plain", device),
         "base_enc": collect(models["base"], tok, rows, "encoded", device)}
    out = {}
    rng = np.random.default_rng(0)
    for target in ("state", "prev"):
        ti = 1 if target == "state" else 2
        print(f"\ntarget = {target}: principal-angle cosines (4 values, 1 = shared direction); joint probe accuracy")
        print("layer   plain~encoded              plain~base(plain text)     encoded~base(enc text)     plain~random   joint->plain joint->enc")
        out[target] = {}
        for l in range(len(A["plain"][0])):
            Xp, yp, Xe, ye = A["plain"][0][l], A["plain"][ti], A["encoded"][0][l], A["encoded"][ti]
            qp, qe = basis(Xp, yp), basis(Xe, ye)
            qbp, qbe = basis(A["base_plain"][0][l], A["base_plain"][ti]), basis(A["base_enc"][0][l], A["base_enc"][ti])
            qr, _ = np.linalg.qr(rng.standard_normal((Xp.shape[1], 4)))
            inst = A["plain"][3]; cut = np.unique(inst)[int(0.8 * len(np.unique(inst)))]
            tr, te = inst < cut, inst >= cut
            Xj = np.concatenate([Xp[tr] - Xp[tr].mean(0), Xe[tr] - Xe[tr].mean(0)])
            yj = np.concatenate([yp[tr], ye[tr]])
            probe = fit_probe(Xj, yj, device)
            jp = eval_probe(probe, Xp[te] - Xp[tr].mean(0), yp[te], device)
            je = eval_probe(probe, Xe[te] - Xe[tr].mean(0), ye[te], device)
            row = {"plain_encoded": cosines(qp, qe), "plain_base": cosines(qp, qbp), "encoded_base": cosines(qe, qbe),
                   "plain_random": cosines(qp, qr), "joint_to_plain": jp, "joint_to_encoded": je}
            out[target][l] = row
            f = lambda v: " ".join(f"{x:.2f}" for x in v)
            print(f"{l:5d}   {f(row['plain_encoded']):24}   {f(row['plain_base']):24}   {f(row['encoded_base']):24}   {f(row['plain_random'][:1]):5}          {jp:.3f}        {je:.3f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
