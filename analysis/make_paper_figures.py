"""
Paper figures.

Colour follows the dataviz reference palette, categorical slots 1-4 in fixed order
(blue / orange / aqua / yellow), validated for light mode: worst adjacent CVD dE 9.1,
normal-vision dE 22.9. Two of those slots sit under 3:1 contrast on a white surface, so
the relief rule applies -- every series is direct-labelled and carries its own marker
shape, meaning identity never depends on colour alone. That also survives greyscale
printing, which a NeurIPS submission has to.

Usage:  python analysis/make_paper_figures.py
Writes: analysis/out/fig_context_scaling.png
        analysis/out/fig_dose_response.png
        analysis/out/fig_diversity.png
        analysis/out/fig_paper_main.png   (2-panel, the one that goes in the paper)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join("analysis", "out")

# categorical slots 1-4, fixed order, never cycled
COLOR = {"reldiff": "#2a78d6", "grdm": "#eb6834",
         "plurel": "#1baf7a", "rdbpfn": "#eda100"}
MARKER = {"reldiff": "o", "grdm": "s", "plurel": "^", "rdbpfn": "D"}
LABEL = {"reldiff": "RelDiff", "grdm": "GRDM", "plurel": "PluRel", "rdbpfn": "RDB-PFN"}
ORDER = ["reldiff", "grdm", "plurel", "rdbpfn"]

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d8d4"

# the two rel-stack tasks rescored at n=2048; the published 30k cells used n=256
REPOWER_30K = {
    ("reldiff", "user-engagement"): 0.864935, ("reldiff", "user-badge"): 0.811736,
    ("plurel", "user-engagement"): 0.652402, ("plurel", "user-badge"): 0.755918,
    ("rdbpfn", "user-engagement"): 0.642216, ("rdbpfn", "user-badge"): 0.749681,
    ("grdm", "user-engagement"): 0.519743, ("grdm", "user-badge"): 0.651820,
}


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(INK2)


def context_curves():
    """Mean AUROC over the six classification tasks, with the corrected 30k point."""
    df = pd.read_csv("rt_benchmark.csv")
    df = df[(df.kind == "clf") & df.auroc.notna()].copy()
    df["auroc_fixed"] = [
        REPOWER_30K.get((r.generator, r.task), r.auroc) if r.ctx_len == 30000 else r.auroc
        for r in df.itertuples()]
    out = {}
    for g in ORDER:
        s = df[df.generator == g].groupby("ctx_len").auroc_fixed.mean()
        out[g] = s.sort_index()
    return out


def panel_context(ax, curves):
    for g in ORDER:
        s = curves[g]
        ax.plot(s.index, s.values, color=COLOR[g], marker=MARKER[g], lw=2,
                markersize=5, label=LABEL[g], clip_on=False, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(LABEL[g], (s.index[-1], s.values[-1]), color=COLOR[g],
                    fontsize=7.5, weight="bold", xytext=(6, 0),
                    textcoords="offset points", va="center")
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 512, 1024, 30000])
    ax.set_xticklabels(["100", "200", "512", "1k", "30k"])
    ax.set_xlabel("in-context length (tokens)", fontsize=8.5, color=INK2)
    ax.set_ylabel("mean AUROC, 6 classification tasks", fontsize=8.5, color=INK2)
    ax.set_title("Every model degrades with context; RelDiff least",
                 fontsize=9.5, color=INK, loc="left", pad=8)
    ax.set_xlim(90, 30000)


def _stagger(ends, min_gap):
    """Nudge end-labels apart so near-identical series stay separately readable.

    Three of the four dose curves land within 0.01 of each other, so their end
    labels would print on top of one another. Sort by value and push each up to at
    least `min_gap` above the previous one, keeping label order = series order.
    """
    out, prev = {}, None
    for g, v in sorted(ends.items(), key=lambda kv: kv[1]):
        y = v if prev is None else max(v, prev + min_gap)
        out[g] = y
        prev = y
    return out


def panel_dose(ax):
    d = pd.read_csv(f"{OUT}/e7_dose_response.csv")
    piv = d.pivot_table(index="model", columns="dose", values="loss")
    ends = {}
    for g in ORDER:
        if g not in piv.index:
            continue
        y = piv.loc[g]
        base = y.loc[0.0]
        ax.plot(y.index * 100, y.values - base, color=COLOR[g], marker=MARKER[g],
                lw=2, markersize=5, label=LABEL[g], clip_on=False, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ends[g] = y.values[-1] - base
    span = max(ends.values()) - min(ends.values())
    for g, ytxt in _stagger(ends, min_gap=span * 0.075).items():
        ax.annotate(LABEL[g], (100, ytxt), color=COLOR[g], fontsize=7.5,
                    weight="bold", xytext=(7, 0), textcoords="offset points",
                    va="center", annotation_clip=False)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("share of FK links corrupted (%)", fontsize=8.5, color=INK2)
    ax.set_ylabel("increase in masked-cell loss", fontsize=8.5, color=INK2)
    ax.set_title("Only RelDiff's loss tracks the corruption",
                 fontsize=9.5, color=INK, loc="left", pad=8)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, 100)


def main():
    os.makedirs(OUT, exist_ok=True)
    curves = context_curves()

    # ---- the figure that goes in the paper
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3))
    panel_context(axes[0], curves)
    panel_dose(axes[1])
    for ax in axes:
        style(ax)
    # one shared legend below both panels: in-panel placement collided with the
    # GRDM curve on the left, and the panels share a series set anyway
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=4, labelcolor=INK2,
               handlelength=1.8, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(pad=1.4, rect=(0, 0.05, 1, 1))
    fig.savefig(f"{OUT}/fig_paper_main.png", dpi=300, bbox_inches="tight",
                facecolor="white")

    # ---- standalone versions
    for name, fn in [("fig_context_scaling", lambda a: panel_context(a, curves)),
                     ("fig_dose_response", panel_dose)]:
        f, a = plt.subplots(figsize=(5.2, 3.2))
        fn(a); style(a)
        a.legend(frameon=False, fontsize=7.5, labelcolor=INK2, handlelength=1.6)
        f.tight_layout(); f.savefig(f"{OUT}/{name}.png", dpi=300,
                                    bbox_inches="tight", facecolor="white")
        plt.close(f)

    # ---- diversity: magnitude across three axes, grouped bars
    dv = pd.read_csv("diversity_rfms.csv")
    keymap = {"RelDiff": "reldiff", "GRDM": "grdm", "PLUREL": "plurel", "RDBPFN": "rdbpfn"}
    dv["key"] = dv.generator.map(keymap)
    axes_names = ["feature_diversity_euclidean", "table_diversity_euclidean",
                  "cross_table_feature_diversity_euclidean"]
    pretty = ["feature", "table", "cross-table"]
    f, a = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(axes_names)); w = 0.19
    for i, g in enumerate(ORDER):
        row = dv[dv.key == g]
        if row.empty:
            continue
        vals = [float(row[c].iloc[0]) for c in axes_names]
        # 2px surface gap between adjacent bars
        a.bar(x + (i - 1.5) * w, vals, width=w * 0.88, color=COLOR[g],
              label=LABEL[g], zorder=3)
        for xi, v in zip(x + (i - 1.5) * w, vals):
            a.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                       fontsize=6.2, color=INK2, xytext=(0, 1.5),
                       textcoords="offset points")
    a.set_xticks(x); a.set_xticklabels(pretty)
    a.set_ylabel("mean pairwise Euclidean distance", fontsize=8.5, color=INK2)
    a.set_title("Output diversity, measured on the generated data",
                fontsize=9.5, color=INK, loc="left", pad=8)
    a.legend(frameon=False, fontsize=7.5, ncol=4, labelcolor=INK2, handlelength=1.2,
             loc="upper center", bbox_to_anchor=(0.5, -0.16))
    style(a)
    f.tight_layout(); f.savefig(f"{OUT}/fig_diversity.png", dpi=300,
                                bbox_inches="tight", facecolor="white")

    print("context-scaling (mean AUROC, corrected 30k):")
    for g in ORDER:
        s = curves[g]
        print(f"  {LABEL[g]:8s} " + "  ".join(f"{c}:{v:.3f}" for c, v in s.items()))
    print(f"\nwrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
