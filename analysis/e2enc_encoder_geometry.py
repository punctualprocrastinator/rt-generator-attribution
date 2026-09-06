"""
E2-enc: typed-encoder and mask-embedding geometry (the E2 of MECH_INTERP_EXPERIMENTS.md).

Not to be confused with the corpus-side predictive-necessity experiment, which this
project also calls E2. This one asks whether the generators' differing value
distributions propagated into different low-level featurization, or whether all four
models converge on the same input geometry -- in which case featurization is
data-insensitive and everything interesting happens deeper.

The number/datetime/boolean encoders are 1 -> 256 maps, so each is just a scaled
direction and can be compared exhaustively: norm (gain) and cosine (direction). The text
and col_name encoders are 384 -> 256, compared by effective rank of their spectrum. The
mask embeddings are single vectors, so we can also ask whether "masked number" sits near
"masked datetime" in the same way across models.

Usage:  python analysis/e2enc_encoder_geometry.py
Writes: analysis/out/e2enc_encoder_stats.csv
        analysis/out/e2enc_pairwise_cosine.csv
        analysis/out/e2enc_maskemb_cosine.csv
        analysis/out/fig_e2enc.png
"""
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CKPT_DIR = "model_checkpoints"
OUT_DIR = os.path.join("analysis", "out")
MODELS = ["reldiff", "grdm", "plurel", "rdbpfn"]
SCALAR_ENC = ["number", "datetime", "boolean"]
WIDE_ENC = ["text", "col_name"]
MASK_TYPES = ["number", "text", "datetime", "boolean"]


def load(name):
    sd = torch.load(f"{CKPT_DIR}/{name}_final.pt", map_location="cpu", weights_only=True)
    return OrderedDict((k.replace("_orig_mod.", ""), v.float()) for k, v in sd.items())


def eff_rank(w):
    """Effective rank = exp(entropy of the normalized singular-value spectrum)."""
    s = torch.linalg.svdvals(w).numpy()
    s = s[s > 0]
    if not len(s):
        return float("nan")
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def cos(a, b):
    a, b = a.flatten().numpy(), b.flatten().numpy()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sds = {m: load(m) for m in MODELS}

    # ---------- per-encoder statistics ----------
    rows = []
    for m in MODELS:
        for e in SCALAR_ENC + WIDE_ENC:
            w = sds[m][f"enc_dict.{e}.weight"]
            b = sds[m].get(f"enc_dict.{e}.bias")
            rec = dict(model=m, encoder=e, shape="x".join(map(str, tuple(w.shape))),
                       weight_norm=float(w.norm()),
                       bias_norm=float(b.norm()) if b is not None else None)
            if e in WIDE_ENC:
                rec["effective_rank"] = eff_rank(w)
                rec["max_singular"] = float(torch.linalg.svdvals(w)[0])
            rows.append(rec)
        for t in MASK_TYPES:
            v = sds[m][f"mask_embs.{t}"]
            rows.append(dict(model=m, encoder=f"mask_emb.{t}",
                             shape="x".join(map(str, tuple(v.shape))),
                             weight_norm=float(v.norm())))
    stats = pd.DataFrame(rows)
    stats.to_csv(f"{OUT_DIR}/e2enc_encoder_stats.csv", index=False)

    # ---------- cross-model agreement per encoder ----------
    prows = []
    for e in SCALAR_ENC + WIDE_ENC:
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                prows.append(dict(encoder=e, pair=f"{a}~{b}",
                                  cosine=cos(sds[a][f"enc_dict.{e}.weight"],
                                             sds[b][f"enc_dict.{e}.weight"])))
    for t in MASK_TYPES:
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                prows.append(dict(encoder=f"mask_emb.{t}", pair=f"{a}~{b}",
                                  cosine=cos(sds[a][f"mask_embs.{t}"],
                                             sds[b][f"mask_embs.{t}"])))
    pair = pd.DataFrame(prows)
    pair.to_csv(f"{OUT_DIR}/e2enc_pairwise_cosine.csv", index=False)

    # ---------- within-model geometry among the four mask embeddings ----------
    mrows = []
    for m in MODELS:
        for i, t1 in enumerate(MASK_TYPES):
            for t2 in MASK_TYPES[i + 1:]:
                mrows.append(dict(model=m, pair=f"{t1}~{t2}",
                                  cosine=cos(sds[m][f"mask_embs.{t1}"],
                                             sds[m][f"mask_embs.{t2}"])))
    memb = pd.DataFrame(mrows)
    memb.to_csv(f"{OUT_DIR}/e2enc_maskemb_cosine.csv", index=False)

    # ---------- report ----------
    print("Encoder weight norms (gain of each typed input map):")
    piv = stats.pivot_table(index="model", columns="encoder", values="weight_norm")
    print(piv.round(3).to_string(), "\n")

    print("Effective rank of the 384->256 projections:")
    wide = stats[stats.encoder.isin(WIDE_ENC)].pivot_table(
        index="model", columns="encoder", values="effective_rank")
    print(wide.round(2).to_string(), "\n")

    print("Cross-model cosine per encoder (1.0 = identical direction):")
    cpiv = pair.pivot_table(index="pair", columns="encoder", values="cosine")
    print(cpiv.round(4).to_string(), "\n")

    print("Within-model cosine among the four mask embeddings:")
    mpiv = memb.pivot_table(index="pair", columns="model", values="cosine")
    print(mpiv.round(3).to_string())

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    order = SCALAR_ENC + WIDE_ENC
    x = np.arange(len(order))
    for i, m in enumerate(MODELS):
        axes[0].bar(x + i * 0.2 - 0.3,
                    [stats[(stats.model == m) & (stats.encoder == e)].weight_norm.iloc[0]
                     for e in order], width=0.2, label=m)
    axes[0].set_xticks(x); axes[0].set_xticklabels(order, fontsize=8)
    axes[0].set_ylabel("weight norm"); axes[0].set_title("Typed encoder gain", fontsize=10)
    axes[0].legend(fontsize=7)

    cm = cpiv.reindex(columns=order)
    im = axes[1].imshow(cm.values, aspect="auto", cmap="viridis")
    axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(order, fontsize=7, rotation=30)
    axes[1].set_yticks(range(len(cm.index))); axes[1].set_yticklabels(cm.index, fontsize=7)
    axes[1].set_title("Cross-model cosine", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    mm = mpiv.reindex(columns=MODELS)
    im2 = axes[2].imshow(mm.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(MODELS))); axes[2].set_xticklabels(MODELS, fontsize=7, rotation=30)
    axes[2].set_yticks(range(len(mm.index))); axes[2].set_yticklabels(mm.index, fontsize=7)
    axes[2].set_title("Mask-embedding geometry", fontsize=10)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("E2-enc: does the corpus reach low-level featurization?", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_e2enc.png", dpi=200)
    print(f"\nwrote 3 CSVs and fig_e2enc.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
