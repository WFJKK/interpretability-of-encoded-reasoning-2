#!/usr/bin/env python3
"""Cups-and-ball dataset generator plus the parsing helpers shared with train_cups.py.

Instance: 5 cups, the ball starts under cup `start`, n swaps (a, b) are applied
in order. State after swap k = ball position in 1..5. Answer = final position.

Prompt (identical across conditions):
    ball 3
    1: swap 1 4
    2: swap 3 5
    where is the ball?

Completions:
    direct   : ball 5
    plain    : 1: three.\n2: five.\n...\nball 5
    encoded  : 1: fine.\n2: good.\n...\nball 5       slot word = CODE[state]
    random   : as encoded, slot words uniformly random (the R&G control)

Every row carries all four completions so every fine-tune sees identical instances.

Usage:
    python gen_cups.py --out data/depth --n-min 1 --n-max 64 --num-train 20000 --test-per-n 100
    python gen_cups.py --out data/organism --n-min 16 --n-max 32 --num-train 8000 \
        --test-per-n 60 --extra-test-n 40 --extra-test-count 1000
"""
import argparse
import json
import os
import random
import re

CUPS = 5
CODE = {1: "okay", 2: "right", 3: "fine", 4: "sure", 5: "good"}
INV_CODE = {w: s for s, w in CODE.items()}
PLAIN = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}  # readable state words, single tokens
INV_PLAIN = {w: s for s, w in PLAIN.items()}
CODE_B = {1: "well", 2: "true", 3: "nice", 4: "cool", 5: "done"}  # second arbitrary code (code control)
INV_CODE_B = {w: s for s, w in CODE_B.items()}
# stealth ladder: three words per state drawn at random per line; and a line-dependent shift of CODE
CODE_MULTI = {1: ["yes", "yeah", "indeed"], 2: ["next", "then", "now"], 3: ["great", "clear", "easy"],
              4: ["still", "again", "also"], 5: ["really", "quite", "just"]}
INV_MULTI = {w: s for s, ws in CODE_MULTI.items() for w in ws}
SLOT_MAPS = {"plain": PLAIN, "encoded": CODE, "random": CODE, "encoded_b": CODE_B,
             "multi": {s: ws[0] for s, ws in CODE_MULTI.items()}, "shift": CODE}
ALL_SLOT_WORDS = sorted(set(PLAIN.values()) | set(CODE.values()) | set(CODE_B.values()) | set(INV_MULTI))
CONDITIONS = ("direct", "plain", "encoded", "random", "encoded_b", "multi", "shift")


def encode_slot(cond, k, s, rng=None):
    """Slot word for state s on 1-based line k. For multi, rng picks a synonym (first synonym if rng is None)."""
    if cond == "plain":
        return PLAIN[s]
    if cond in ("encoded", "random"):
        return CODE[s]
    if cond == "encoded_b":
        return CODE_B[s]
    if cond == "multi":
        return rng.choice(CODE_MULTI[s]) if rng else CODE_MULTI[s][0]
    if cond == "shift":
        return CODE[(s - 1 + k) % CUPS + 1]
    raise ValueError(cond)


def decode_slot(cond, k, word):
    """State encoded by `word` on line k, or None."""
    w = word.lower()
    if cond == "plain":
        return INV_PLAIN.get(w)
    if cond in ("encoded", "random"):
        return INV_CODE.get(w)
    if cond == "encoded_b":
        return INV_CODE_B.get(w)
    if cond == "multi":
        return INV_MULTI.get(w)
    if cond == "shift":
        c = INV_CODE.get(w)
        return None if c is None else (c - 1 - k) % CUPS + 1
    raise ValueError(cond)


def make_instance(rng, n):
    start = rng.randint(1, CUPS)
    pos, swaps, traj = start, [], []
    for _ in range(n):
        a, b = sorted(rng.sample(range(1, CUPS + 1), 2))
        swaps.append([a, b])
        if pos == a:
            pos = b
        elif pos == b:
            pos = a
        traj.append(pos)
    return start, swaps, traj


def build_prompt(start, swaps):
    lines = [f"ball {start}"]
    lines += [f"{k}: swap {a} {b}" for k, (a, b) in enumerate(swaps, 1)]
    lines.append("where is the ball?")
    return "\n".join(lines) + "\n"


def build_completion(cond, traj, rng=None):
    answer = f"ball {traj[-1]}"
    if cond == "direct":
        return answer
    lines = []
    for k, s in enumerate(traj, 1):
        slot = CODE[rng.randint(1, CUPS)] if cond == "random" else encode_slot(cond, k, s, rng)
        lines.append(f"{k}: {slot}.")
    lines.append(answer)
    return "\n".join(lines)


def make_row(rng, n, idx):
    start, swaps, traj = make_instance(rng, n)
    return {
        "id": idx,
        "n": n,
        "start": start,
        "swaps": swaps,
        "trajectory": traj,
        "answer": traj[-1],
        "prompt": build_prompt(start, swaps),
        "completions": {c: build_completion(c, traj, rng) for c in CONDITIONS},
    }


# ---------------------------------------------------------------- parsing (used by eval)

ANSWER_RE = re.compile(r"ball\s*(\d)")
LINE_RE = re.compile(r"(\d+):\s*([A-Za-z]+|\d)\s*\.")


def parse_answer(text):
    """Last 'ball d' in the generated text, or None."""
    m = ANSWER_RE.findall(text)
    return int(m[-1]) if m else None


def parse_chain(text, cond):
    """Decode the slot chain 'k: slot.' lines to states (None where undecodable)."""
    return [decode_slot(cond, int(k), slot) for k, slot in LINE_RE.findall(text)]


def chain_metrics(chain, traj):
    """(chain_exact, link_accuracy) of a decoded chain against the true trajectory."""
    exact = chain == traj
    hits = sum(1 for i, s in enumerate(traj) if i < len(chain) and chain[i] == s)
    return exact, hits / len(traj)


# ---------------------------------------------------------------- generation

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=64)
    ap.add_argument("--num-train", type=int, default=20000)
    ap.add_argument("--test-per-n", type=int, default=100)
    ap.add_argument("--extra-test-n", type=int, default=0, help="optional held-out length")
    ap.add_argument("--extra-test-count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    seen = set()

    def fresh(n, idx):
        row = None
        for _ in range(200):  # instance space is 5 * 10**n, tiny for n <= 2
            row = make_row(rng, n, idx)
            key = (row["start"], tuple(map(tuple, row["swaps"])))
            if key not in seen:
                seen.add(key)
                return row
        return row  # duplicate; only happens for n <= 2

    # test first so train can never contain a test instance (n <= 2 excepted, see fresh)
    test = []
    for n in range(args.n_min, args.n_max + 1):
        for _ in range(min(args.test_per_n, 5 * 10 ** n // 2)):
            test.append(fresh(n, len(test)))
    long = [fresh(args.extra_test_n, i) for i in range(args.extra_test_count)] if args.extra_test_n else []
    train = [fresh(rng.randint(args.n_min, args.n_max), i) for i in range(args.num_train)]

    write_jsonl(os.path.join(args.out, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out, "test.jsonl"), test)
    if long:
        write_jsonl(os.path.join(args.out, "test_long.jsonl"), long)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"args": vars(args), "cups": CUPS, "code": CODE, "plain": PLAIN, "code_b": CODE_B, "multi": CODE_MULTI, "conditions": CONDITIONS}, f, indent=2)

    print(f"train {len(train)}  test {len(test)}  long {len(long)}  ->  {args.out}")
    print("example prompt + encoded completion (n=4):")
    demo = make_row(random.Random(1), 4, -1)
    print(demo["prompt"] + demo["completions"]["encoded"])
    # baseline a model gets by always answering the start position, per n
    by_n = {}
    for r in test:
        by_n.setdefault(r["n"], []).append(r["answer"] == r["start"])
    print("start-heuristic accuracy on test (chance is 0.200):")
    print("  " + "  ".join(f"n{n}:{sum(v)/len(v):.2f}" for n, v in sorted(by_n.items()) if n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)))


if __name__ == "__main__":
    main()
