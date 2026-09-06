"""
E8-w: is the neighbour-attention specialization carried by a few heads or spread
across all of them?

The shortlist asks whether RelDiff's relational reliance is a small circuit or a
distributed property. The behavioural version needs a forward pass, but the
weight-space version does not: each attention projection is [n_heads*head_dim,
d_model], so a head owns a contiguous slice and its movement can be measured on
its own.

q_norm has shape (32,) against d_model 256, so there are 8 heads of 32.

For every (model, block, stream, head) we take the deviation from the 4-model
mean and report its norm, then summarize how unevenly that deviation is spread
over heads with a Gini coefficient and a top-2-of-8 mass share. A concentrated
profile means a circuit worth naming; a flat one means the corpus moved the whole
stream.

Usage:  python analysis/e8w_head_concentration.py
Writes: analysis/out/e8w_head_deviation.csv
        analysis/out/e8w_concentration.csv
        analysis/out/fig_e8w_heads.png
"""
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CKPT_DIR = "model_checkpoints"
OUT_DIR = os.path.join("analysis", "out")
MODELS = ["grdm", "plurel", "rdbpfn", "reldiff"]
STREAMS = ["col", "feat", "nbr"]
N_HEADS = 8
HEAD_DIM = 32


def load(name):
    sd = torch.load(f"{CKPT_DIR}/{name}_final.pt", map_location="cpu", weights_only=True)
    return OrderedDict((k.replace("_orig_mod.", ""), v.float()) for k, v in sd.items())


def head_slice(tensor, proj, h):
    """Rows for q/k/v (out = heads*dim); columns for the output projection."""
    lo, hi = h * HEAD_DIM, (h + 1) * HEAD_DIM
    return tensor[:, lo:hi] if proj == "wo" else tensor[lo:hi, :]


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() <= 0:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sds = {m: load(m) for m in MODELS}
    keys = list(sds[MODELS[0]].keys())
    blocks = sorted({int(m.group(1)) for m in
                     (re.match(r"blocks\.(\d+)\.", k) for k in keys) if m})

    mean = {k: sum(sds[m][k] for m in MODELS) / len(MODELS) for k in keys}

    rows = []
    for m in MODELS:
        for b in blocks:
            for s in STREAMS:
                for h in range(N_HEADS):
                    acc = 0.0
                    for proj in ["wq", "wk", "wv", "wo"]:
                        k = f"blocks.{b}.attns.{s}.{proj}.weight"
                        dev = sds[m][k] - mean[k]
                        acc += float(head_slice(dev, proj, h).pow(2).sum())
                    rows.append(dict(model=m, block=b, stream=s, head=h,
                                     deviation_l2=float(np.sqrt(acc))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/e8w_head_deviation.csv", index=False)

    # concentration of deviation across the 8 heads, per model and stream
    crows = []
    for m in MODELS:
        for s in STREAMS:
            g = df[(df.model == m) & (df.stream == s)]
            per_head = g.groupby("head").deviation_l2.apply(
                lambda v: float(np.sqrt((v ** 2).sum()))).values
            order = np.sort(per_head)[::-1]
            crows.append(dict(
                model=m, stream=s,
                total_l2=float(np.sqrt((per_head ** 2).sum())),
                gini=gini(per_head),
                top1_share=float(order[0] / per_head.sum()),
                top2_share=float(order[:2].sum() / per_head.sum()),
                max_over_min=float(order[0] / order[-1]),
            ))
    cdf = pd.DataFrame(crows)
    cdf.to_csv(f"{OUT_DIR}/e8w_concentration.csv", index=False)

    print("Per-head concentration of deviation (8 heads; uniform => gini 0, "
          "top2 share 0.25)\n")
    for s in STREAMS:
        print(f"  stream = {s}")
        print(f"    {'model':9s} {'gini':>7s} {'top1':>7s} {'top2':>7s} {'max/min':>8s}")
        for m in MODELS:
            r = cdf[(cdf.model == m) & (cdf.stream == s)].iloc[0]
            print(f"    {m:9s} {r.gini:7.3f} {r.top1_share:7.3f} "
                  f"{r.top2_share:7.3f} {r.max_over_min:8.2f}")
        print()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, s in zip(axes, STREAMS):
        for m in MODELS:
            g = df[(df.model == m) & (df.stream == s)]
            per_head = g.groupby("head").deviation_l2.apply(
                lambda v: float(np.sqrt((v ** 2).sum())))
            ax.plot(per_head.index, per_head.values, marker="o", label=m, lw=1.4)
        ax.set_title(f"{s}-attention", fontsize=10)
        ax.set_xlabel("head")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("deviation from 4-model mean")
    axes[0].legend(fontsize=8)
    fig.suptitle("Is the corpus imprint carried by a few heads?", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_e8w_heads.png", dpi=200)
    print(f"wrote {OUT_DIR}/e8w_head_deviation.csv, e8w_concentration.csv, fig_e8w_heads.png")


if __name__ == "__main__":
    main()
