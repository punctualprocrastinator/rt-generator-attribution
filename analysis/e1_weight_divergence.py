"""
E1: weight-space divergence across the four RT checkpoints.

Recomputes the per-stream deviation table reported in the paper, adds a
per-block breakdown, and tests the shared-initialization assumption that
E1's interpretation depends on.

Usage:  python analysis/e1_weight_divergence.py
Writes: analysis/out/e1_results.json
        analysis/out/fig_e1_stream_depth.png
        analysis/out/fig_e1_stream_bars.png
"""
import json
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CKPT_DIR = "model_checkpoints"
OUT_DIR = os.path.join("analysis", "out")
MODELS = ["grdm", "plurel", "rdbpfn", "reldiff"]
STREAMS = ["encoders", "col", "feat", "nbr", "ffn"]


def load(name):
    sd = torch.load(f"{CKPT_DIR}/{name}_final.pt", map_location="cpu", weights_only=True)
    return OrderedDict((k.replace("_orig_mod.", ""), v.float()) for k, v in sd.items())


def stream_of(key):
    """Map a parameter name to one of the reported streams, or None to ignore."""
    if key.startswith("enc_dict."):
        return "encoders"
    if key.startswith("dec_dict."):
        return "decoders"
    m = re.match(r"blocks\.(\d+)\.attns\.(col|feat|nbr)\.", key)
    if m:
        return m.group(2)
    if re.match(r"blocks\.(\d+)\.ffn\.", key):
        return "ffn"
    if key.startswith("norm") or ".norms." in key:
        return "norms"
    if key.startswith("mask_embs"):
        return "mask_embs"
    return None


def block_of(key):
    m = re.match(r"blocks\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def flat(sd, keys):
    return np.concatenate([sd[k].numpy().ravel() for k in keys]) if keys else np.array([])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sds = {m: load(m) for m in MODELS}
    keys = list(sds[MODELS[0]].keys())
    for m in MODELS:
        assert list(sds[m].keys()) == keys, f"{m} has a different parameter set"

    results = {"n_tensors": len(keys),
               "n_params": int(sum(v.numel() for v in sds[MODELS[0]].values()))}

    # ---------- shared-initialization evidence ----------
    # If the arms share an init and moved little, whole-vector cosines sit near 1.
    # A sharper test: tensors that are bitwise identical across all four models
    # can only be explained by a shared starting point (training would perturb
    # them differently), so we count them and their parameter mass.
    full = {m: flat(sds[m], keys) for m in MODELS}
    raw_cos = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            raw_cos[f"{a}~{b}"] = float(
                np.dot(full[a], full[b]) / (np.linalg.norm(full[a]) * np.linalg.norm(full[b])))
    identical = [k for k in keys
                 if all(torch.equal(sds[MODELS[0]][k], sds[m][k]) for m in MODELS[1:])]
    results["shared_init_evidence"] = {
        "raw_pairwise_cosine": raw_cos,
        "n_bitwise_identical_tensors": len(identical),
        "bitwise_identical_tensors": identical[:40],
        "identical_param_mass": int(sum(sds[MODELS[0]][k].numel() for k in identical)),
    }

    # ---------- deviation from the 4-model mean (init proxy) ----------
    mean = {k: sum(sds[m][k] for m in MODELS) / len(MODELS) for k in keys}
    dev = {m: OrderedDict((k, sds[m][k] - mean[k]) for k in keys) for m in MODELS}

    by_stream = {}
    for s in set(stream_of(k) for k in keys) - {None}:
        ks = [k for k in keys if stream_of(k) == s]
        by_stream[s] = {m: float(np.linalg.norm(flat(dev[m], ks))) for m in MODELS}
    results["deviation_norm_by_stream"] = by_stream

    dvec = {m: flat(dev[m], keys) for m in MODELS}
    results["deviation_norm_total"] = {m: float(np.linalg.norm(dvec[m])) for m in MODELS}
    dev_cos = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            dev_cos[f"{a}~{b}"] = float(
                np.dot(dvec[a], dvec[b]) / (np.linalg.norm(dvec[a]) * np.linalg.norm(dvec[b])))
    results["deviation_cosine"] = dev_cos
    # four mean-centred vectors must average -1/3; that is the "unrelated" baseline
    results["deviation_cosine_baseline"] = -1.0 / (len(MODELS) - 1)

    # ---------- per-block, per-stream ----------
    blocks = sorted({block_of(k) for k in keys} - {None})
    per_block = {}
    for s in ["col", "feat", "nbr", "ffn"]:
        per_block[s] = {}
        for b in blocks:
            ks = [k for k in keys if stream_of(k) == s and block_of(k) == b]
            per_block[s][b] = {m: float(np.linalg.norm(flat(dev[m], ks))) for m in MODELS}
    results["deviation_norm_by_block"] = per_block

    # Parameter-count normalised version. Raw norms scale with tensor size, so the
    # FFN dominates by mass alone; dividing by sqrt(n) makes streams comparable.
    norm_by_stream = {}
    for s in by_stream:
        ks = [k for k in keys if stream_of(k) == s]
        n = sum(sds[MODELS[0]][k].numel() for k in ks)
        norm_by_stream[s] = {m: by_stream[s][m] / np.sqrt(n) for m in MODELS}
    results["deviation_rms_by_stream"] = norm_by_stream

    with open(f"{OUT_DIR}/e1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------- CSV exports ----------
    import csv as _csv

    def _write(path, header, rows):
        with open(path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    stream_order = ["encoders", "decoders", "col", "feat", "nbr", "ffn", "norms", "mask_embs"]
    present = [s for s in stream_order if s in by_stream]
    _write(f"{OUT_DIR}/e1_stream_deviation.csv",
           ["model", "stream", "deviation_l2", "deviation_rms"],
           [[m, s, by_stream[s][m], norm_by_stream[s][m]] for m in MODELS for s in present])

    _write(f"{OUT_DIR}/e1_block_deviation.csv",
           ["model", "stream", "block", "deviation_l2"],
           [[m, s, b, per_block[s][b][m]]
            for m in MODELS for s in ["col", "feat", "nbr", "ffn"] for b in blocks])

    _write(f"{OUT_DIR}/e1_pairwise.csv",
           ["pair", "raw_cosine", "deviation_cosine", "deviation_cosine_baseline"],
           [[k, raw_cos[k], dev_cos[k], results["deviation_cosine_baseline"]] for k in raw_cos])

    _write(f"{OUT_DIR}/e1_model_totals.csv",
           ["model", "deviation_l2_total"],
           [[m, results["deviation_norm_total"][m]] for m in MODELS])

    _write(f"{OUT_DIR}/e1_shared_init.csv",
           ["metric", "value"],
           [["n_tensors_total", results["n_tensors"]],
            ["n_params_total", results["n_params"]],
            ["n_bitwise_identical_tensors", len(identical)],
            ["identical_param_mass", results["shared_init_evidence"]["identical_param_mass"]],
            ["min_raw_pairwise_cosine", min(raw_cos.values())],
            ["max_raw_pairwise_cosine", max(raw_cos.values())]])

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True)
    for ax, m in zip(axes, MODELS):
        mat = np.array([[per_block[s][b][m] for b in blocks] for s in ["col", "feat", "nbr"]])
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels(blocks, fontsize=7)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["col", "feat", "nbr"], fontsize=8)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("block", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Deviation from 4-model mean, by attention stream and depth", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_e1_stream_depth.png", dpi=200)

    fig2, ax = plt.subplots(figsize=(6.2, 3.2))
    order = ["encoders", "col", "feat", "nbr", "ffn"]
    x = np.arange(len(order))
    for i, m in enumerate(MODELS):
        ax.bar(x + i * 0.2 - 0.3, [by_stream[s][m] for s in order], width=0.2, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("deviation L2 norm")
    ax.legend(fontsize=8)
    ax.set_title("Where each corpus moved the weights", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(f"{OUT_DIR}/fig_e1_stream_bars.png", dpi=200)

    # ---------- console summary ----------
    print(f"tensors={results['n_tensors']}  params={results['n_params']/1e6:.2f}M")
    print("\nraw pairwise cosine (whole parameter vector):")
    for k, v in raw_cos.items():
        print(f"  {k:20s} {v:.5f}")
    print(f"\nbitwise-identical tensors across all 4: {len(identical)}"
          f"  ({results['shared_init_evidence']['identical_param_mass']} params)")
    for k in identical[:10]:
        print("   ", k)

    print("\ndeviation norm by stream (paper Table 1 columns):")
    hdr = ["encoders", "col", "feat", "nbr", "ffn"]
    print(f"  {'model':9s}" + "".join(f"{h:>10s}" for h in hdr))
    for m in MODELS:
        print(f"  {m:9s}" + "".join(f"{by_stream[h][m]:10.2f}" for h in hdr))

    print("\nper-parameter RMS deviation by stream (size-normalised):")
    print(f"  {'model':9s}" + "".join(f"{h:>10s}" for h in hdr))
    for m in MODELS:
        print(f"  {m:9s}" + "".join(f"{norm_by_stream[h][m]*1e3:10.3f}" for h in hdr))
    print("  (x1e-3)")

    print("\ntotal deviation norm:", {m: round(results['deviation_norm_total'][m], 2) for m in MODELS})
    print("deviation cosines (baseline -0.333):")
    for k, v in dev_cos.items():
        print(f"  {k:20s} {v:+.3f}")
    print(f"\nwrote {OUT_DIR}/e1_results.json and 2 figures")


if __name__ == "__main__":
    main()
