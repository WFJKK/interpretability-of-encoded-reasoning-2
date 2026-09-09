# Interpretability of encoded reasoning, part 2

A minimal, load-bearing model organism for encoded reasoning in the sense of Roger and Greenblatt (2023), built on a task simple enough for circuit-level interpretability.

## Task

Five cups, a ball under one of them, n swaps of two cups. The state after swap k is the ball's position (1 to 5). The prompt lists the start position and the swaps; the answer is the final position.

    ball 3
    1: swap 1 4
    2: swap 3 5
    3: swap 2 5
    where is the ball?

Trajectory 3, 5, 2. Answer 2.

## Conditions

Four fine-tunes of the same base model on identical instances, differing only in the completion:

    direct    ball 2
    plain     1: three.  2: five.  3: two.  ball 2
    encoded   1: fine.  2: good.  3: right.  ball 2
    random    as encoded with random slot words (control)

The slot word after index k is the ball's position after swap k under a fixed map (1 okay, 2 right, 3 fine, 4 sure, 5 good; plain uses one to five). The index is a counter the model uses to look up swap k in the prompt and carries no state. In the encoded condition the intermediate results of the computation appear only as this word choice.

## Why the slot words are load-bearing

Transpositions on five points generate S5, which is non-solvable, so no constant-depth circuit tracks the ball; a model traces at most some fixed amount in one forward pass. Training on n well above that forces the state through the emitted token.

Measured first with the direct condition trained on n uniform in 1..64 (20k examples, 3 epochs). Accuracy against the 0.2 chance level:

    n              1     2     3     4     5     6     7     8     9    10+
    GPT-2 small   1.00  0.83  0.63  0.48  0.35  0.26  0.21  0.23  0.16  0.2
    Qwen2.5-0.5B  1.00  1.00  0.96  0.78  0.70  0.46  0.43  0.31  0.27  0.2

Binned by the number of times the ball actually moves (it moves on 40% of swaps), the capacity is a fixed number of moves, independent of n:

    moves          0     1     2     3+
    GPT-2 small   1.00  0.61  0.18  0.2
    Qwen2.5-0.5B  0.99  1.00  0.68  0.2

The organism trains on n in 16..32 (8k examples, 3 epochs, full fine-tune), where the chance of two or fewer moves is under 2%, with 1000 instances at n = 40 held out.

## Results

Free generation on 1020 held-out instances (n = 16..32) and 1000 at n = 40:

| model | condition | acc n=16..32 | chain exact | acc n=40 | chain exact n=40 |
|---|---|---|---|---|---|
| gpt2 | direct | 0.206 | - | 0.195 | - |
| gpt2 | plain | 1.000 | 1.000 | 0.948 | 0.947 |
| gpt2 | encoded | 1.000 | 1.000 | 1.000 | 1.000 |
| gpt2 | random | 0.210 | - | 0.197 | - |
| qwen2.5-0.5b | direct | 0.205 | - | 0.195 | - |
| qwen2.5-0.5b | plain | 1.000 | 1.000 | 0.984 | 0.984 |
| qwen2.5-0.5b | encoded | 1.000 | 1.000 | 0.209 | 0.012 |
| qwen2.5-0.5b | random | 0.204 | - | 0.191 | - |

Battery on the plain and encoded models (battery.py). Prefixes are teacher-forced from the true trajectory. no_cot: prompt then `ball`. neutralise: every slot word replaced by `ok`, everything else intact. scramble: slot words permuted; reported is how often the answer follows the last slot word. edit: one middle slot word replaced by a wrong code word; "follows" is agreement with the trajectory recomputed from the edited state, "ignores" with the original (the two never coincide).

| model | trained on | full | no_cot | neutralise | scramble follows last slot | edit follows | edit ignores | link follow |
|---|---|---|---|---|---|---|---|---|
| gpt2 | plain | 1.000 | 0.185 | 0.178 | 1.000 | 1.000 | 0.000 | 1.000 |
| gpt2 | encoded | 1.000 | 0.203 | 0.205 | 1.000 | 1.000 | 0.000 | 1.000 |
| qwen2.5-0.5b | plain | 1.000 | 0.217 | 0.185 | 1.000 | 1.000 | 0.000 | 1.000 |
| qwen2.5-0.5b | encoded | 1.000 | 0.192 | 0.205 | 1.000 | 1.000 | 0.000 | 1.000 |

Reading: the encoded models answer perfectly, fall to chance the moment the slot words are removed (with the rest of the completion intact), read the answer off the last slot word whatever it is, and propagate an edited state through every later link with no recovery. The random-slot control matches direct. This is encoded reasoning that is load-bearing by construction, obtained by plain SFT with nothing in the prompt beyond the problem instance.

Notes. Qwen encoded does not length-generalise: at n = 40 it emits `ball` after line 30 in 98% of cases (training sequences with a line 30 always end within two more lines), so its n = 40 row is a stopping-rule failure, not a state-tracking one. GPT-2 encoded generalises perfectly. The plain models show sporadic late link errors at n = 40.

## Layout

    gen_cups.py        dataset generator and parsing helpers
    train_cups.py      tokcheck / train / eval (HF Trainer, resumable, device auto-detected)
    battery.py         no_cot / neutralise / scramble / edit battery for a plain or encoded model
    run_depth.sh       depth measurement for one base model (direct, n in 1..64)
    run_organism.sh    the four organism fine-tunes plus evals for one base model
    summary.py         regenerates the tables above from results/
    results/           committed result JSONs (per-example records included)
    data/, runs/       generated data and models (not committed; data regenerates from the seed)

## Quick start

    pip install -r requirements.txt
    bash run_depth.sh gpt2
    bash run_organism.sh gpt2
    python battery.py --model-path runs/organism-gpt2-encoded/final --condition encoded --data data/organism --out results/battery_gpt2_encoded.json
    python summary.py

Training notes: transformers 5 (warmup_steps takes a float ratio); Qwen2.5 splits digits, hence number words in the plain slot; two fine-tunes do not fit side by side on a 40 GB card.

## Status

Organism established on GPT-2 small and Qwen2.5-0.5B. Next: mechanistic analysis on the GPT-2 plain and encoded models.

## Models

Final weights (private, Hugging Face): `WFJKK/cups-organism-<model>-<condition>` for model in gpt2, qwen2.5-0.5b and condition in direct, plain, encoded, random; the depth-run direct models are `WFJKK/cups-depth-gpt2` and `WFJKK/cups-depth-qwen2.5-0.5b`. Load with `AutoModelForCausalLM.from_pretrained(repo)`; the GPT-2 ones load into TransformerLens via `HookedTransformer.from_pretrained("gpt2", hf_model=...)`.

## Interpretability (GPT-2 small plain and encoded models)

Scripts: interp_boundary.py (probe transfer, weight diff, state-aligned row transplant), interp_circuit.py (attention and activation patching of the per-link update), interp_subspace.py (principal angles between state subspaces), battery.py with --conditions horizon. Results in results/interp_*.json and results/horizon_gpt2_*.json. Activations are read at the token before each slot word, where the slot word is predicted.

Where the code lives. The five slot-word rows of the tied embedding matrix barely move in either fine-tune (delta norm 0.16 plain, 0.29 encoded, against 2 to 3 per layer block), and putting the base model's rows back changes nothing (1.000 on both models), while putting the other model's rows in breaks the recipient (0.324, 0.159). Each fine-tune adapted the internal weights to read and write the pretrained vectors of its own five words. The code is not a relabelling at the embedding boundary.

State representations. Linear probes for the state trained on one model are at chance on the other at every layer (only encoded-trained probes on plain reach 0.65 to 0.76 at layers 11 and 12). Principal cosines between the two models' state subspaces at layers 6 to 11 are 0.24 to 0.38 for the first direction and 0.0 to 0.2 for the rest (random baseline 0.10). Plain's state subspace lies largely within the base model's representation of the number words (cosines 0.8 to 0.9 at layers 9 to 11); encoded's overlaps its base reference less (0.4 to 0.6). A joint probe trained on both models reads both at 1.000, by using the union of two different subspaces, not a shared one.

Circuit. Both models read swap k with the same heads (L5H1, L5H5, L6H9, L7H2, L7H10, attention 0.9 to 1.0 on prompt line k) and carry the previous state with overlapping heads (L7H8 0.94 and L6H1 0.75 in encoded; L8H11, L7H8, L8H1, L6H1 plus early heads L2H2, L2H3, L3H11, L4H3 in plain). Patching the previous slot word: restoring its token identity at layer 0 recovers 100% of the effect; the state stays in that position's residual until layer 6 (encoded) or leaks out gradually from layer 3 (plain), crosses to the ':' position between layers 7 and 9, and the intermediate tokens carry nothing. The transposition is computed in the late MLPs (8 to 11 encoded, 10 to 11 plain), distributed rather than in one component; no single head recovers more than 0.30.

Residual horizon. With the last m slot words neutralised, the encoded model is at chance already at m = 1 (0.205 for all m): the state does not survive a single step without a token. The plain model gives 0.59, 0.30, then chance, which equals the probability that the ball did not move in the last one or two swaps: it falls back on the last readable number word, and the encoded model does not fall back on the last readable code word.

Reading. Encoded reasoning here is the same computation with a different internal state coding, not the same coding with different output words. A white-box monitor trained on the readable model does not detect the encoded state; one has to be trained on the encoded model, or on both.
