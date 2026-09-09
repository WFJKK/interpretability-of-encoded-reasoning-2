#!/usr/bin/env python3
"""Pairwise representation comparison across any set of organism models (seed and code controls).

    python interp_pairs.py --data data/organism --out results/interp_pairs.json \
        plain0=WFJKK/cups-organism-gpt2-plain:plain  plain1=runs/organism-gpt2-plain-s1/final:plain \
        enc0=WFJKK/cups-organism-gpt2-encoded:encoded  enc1=runs/organism-gpt2-encoded-s1/final:encoded \
        encB=runs/organism-gpt2-encoded_b-s0/final:encoded_b

Each spec is label=model_path_or_repo:condition; the condition selects the completion text the model is run
on. Activations are read at the ':' before each slot word. For every ordered pair and layer: strict transfer
accuracy of a state probe trained on the first model and applied unchanged to the second. For every unordered
pair: the first two principal cosines between the two 4-dim state subspaces (random baseline about 0.1).
Within-model held-out accuracy is on the diagonal.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from interp_boundary import collect, eval_probe, fit_probe
from interp_subspace import basis, cosines
from train_cups import describe_device, load_model, load_tokenizer, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="label=path_or_repo:condition")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--layers", default="6,7,8,9,10,11,12", help="layers to print (all are saved)")
    args = ap.parse_args()

    device, _ = describe_device()
    tok = load_tokenizer("gpt2")
    rows = sorted(read_jsonl(os.path.join(args.data, "test.jsonl")), key=lambda r: r["id"])[: args.n_instances]
    specs = []
    for sp in args.specs:
        label, rest = sp.split("=", 1)
        path, cond = rest.rsplit(":", 1)
        specs.append((label, path, cond))

    acts = {}
    for label, path, cond in specs:
        model = load_model(path, torch.float32).to(device).eval()
        acts[label] = collect(model, tok, rows, cond, device)
        del model
        torch.cuda.empty_cache()
        print(f"collected {label} ({path}, {cond})", flush=True)

    labels = [s[0] for s in specs]
    n_layers = len(acts[labels[0]][0])
    inst = acts[labels[0]][3]
    cut = np.unique(inst)[int(0.8 * len(np.unique(inst)))]
    tr, te = inst < cut, inst >= cut

    transfer = {l: {} for l in range(n_layers)}
    cos = {l: {} for l in range(n_layers)}
    for l in range(n_layers):
        probes = {a: fit_probe(acts[a][0][l][tr], acts[a][1][tr], device) for a in labels}
        for a in labels:
            for b in labels:
                transfer[l][f"{a}->{b}"] = eval_probe(probes[a], acts[b][0][l][te], acts[b][1][te], device)
        bases = {a: basis(acts[a][0][l], acts[a][1]) for a in labels}
        for a, b in itertools.combinations(labels, 2):
            cos[l][f"{a}~{b}"] = cosines(bases[a], bases[b])

    show = [int(x) for x in args.layers.split(",")]
    for l in show:
        print(f"\nlayer {l}: strict probe transfer, rows = trained on, columns = applied to (chance 0.2)")
        print("         " + "".join(f"{b:>8}" for b in labels))
        for a in labels:
            print(f"{a:>8} " + "".join(f"{transfer[l][f'{a}->{b}']:8.2f}" for b in labels))
        print(f"layer {l}: first two principal cosines of the state subspaces")
        print("   " + "   ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in cos[l].items()))

    with open(args.out, "w") as f:
        json.dump({"specs": specs, "transfer": transfer, "cosines": cos}, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
