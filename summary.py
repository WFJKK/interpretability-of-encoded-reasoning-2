#!/usr/bin/env python3
"""Print the organism results as markdown tables from results/*.json."""
import json, os
TAGS, CONDS = ["gpt2", "qwen2.5-0.5b"], ["direct", "plain", "encoded", "random"]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def mean(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]; return f"{sum(v)/len(v):.3f}" if v else "-"
print("| model | condition | acc n=16..32 | chain exact | acc n=40 | chain exact n=40 |\n|---|---|---|---|---|---|")
for t in TAGS:
    for c in CONDS:
        r, l = load(f"results/organism_{t}_{c}.json"), load(f"results/organism_{t}_{c}_n40.json")
        if r and l:
            print(f"| {t} | {c} | {mean(r['records'],'correct')} | {mean(r['records'],'chain_exact')} | {mean(l['records'],'correct')} | {mean(l['records'],'chain_exact')} |")
print("\n| model | trained on | full | no_cot | neutralise | scramble follows last slot | edit follows | edit ignores | link follow |\n|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    for c in ("plain", "encoded"):
        b = load(f"results/battery_{t}_{c}.json")
        if not b: continue
        R = b["results"]
        print(f"| {t} | {c} | {R['full']['overall']['correct']:.3f} | {R['no_cot']['overall']['correct']:.3f} | {R['neutralise']['overall']['correct']:.3f} | "
              f"{R['scramble']['overall']['follows_last_slot']:.3f} | {R['edit']['overall']['follows']:.3f} | {R['edit']['overall']['ignores']:.3f} | {R['edit']['overall']['link_follow_acc']:.3f} |")
