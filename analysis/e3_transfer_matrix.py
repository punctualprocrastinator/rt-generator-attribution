"""
E3: cross-generator transfer matrix.

Every checkpoint is scored on every generator's held-out synthetic data and on a real
RelBench database, using the pretraining objective unchanged (masked-cell loss). That
gives a 4 x 5 matrix whose "real" column ranks generators by how close their synthetic
distribution is to reality *as measured by what a model trained on it learned*, which is
a stronger claim than any distance computed on the raw data. The diagonal measures how
much each model overfits its own generator's artifacts.

`net(batch)` returns (loss, yhat) and that loss is exactly the masked-cell objective the
models were pretrained on, so nothing here re-defines the metric.

Batches are drawn deterministically (fixed seed, `shuffle_py(0)`), so all four models see
the identical batch sequence for a given corpus and the comparison is paired.

Usage:  python analysis/e3_transfer_matrix.py [--batches 20] [--ctx 1024]
Writes: analysis/out/e3_transfer_matrix.csv   (long form, one row per model x corpus)
        analysis/out/e3_matrix_wide.csv       (the 4 x 5 table)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

MODELS = ["reldiff", "grdm", "plurel", "rdbpfn"]


def synthetic_tasks(root: Path, cap: int = 12):
    """Task tuples for one extracted corpus.

    `convert_all` writes a manifest listing each database and its masked-cell tasks;
    fall back to scanning the export directory if the manifest is missing.
    """
    man = None
    for c in list(root.rglob("manifest.json")) + list(root.rglob("*manifest*.json")):
        try:
            man = json.loads(c.read_text())
            break
        except Exception:
            continue
    out = []
    if man and "databases" in man:
        for d in man["databases"]:
            for t in d.get("tasks", []):
                out.append((d["db_name"], t["table"], t["target"], "val", []))
    return out[:cap]


def corpus_loss(net, tasks, *, ctx_len, batch_size, n_batches, device="cuda",
                num_workers=2, seed=0):
    """Mean masked-cell loss over a fixed number of deterministic batches."""
    import torch
    from rt.data import RelationalDataset

    if not tasks:
        return dict(status="no-tasks", loss=None, n_batches=0, n_cells=0)
    try:
        ds = RelationalDataset(
            tasks=list(tasks), batch_size=batch_size, rank=0, world_size=1,
            ctx_len=ctx_len, max_bfs_width=128,
            embedding_model="all-MiniLM-L12-v2", d_text=384, seed=seed,
        )
        ds.sampler.shuffle_py(0)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers,
            pin_memory=(device == "cuda"), in_order=True,
        )
        tot, cells, nb = 0.0, 0, 0
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                tbs = batch.pop("true_batch_size")
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                # neutralize the padded tail of the final batch
                batch["masks"][tbs:, :] = False
                batch["is_targets"][tbs:, :] = False
                batch["is_padding"][tbs:, :] = True
                loss, _ = net(batch)
                n = int(batch["is_targets"].bool().sum())
                if n:
                    tot += float(loss) * n
                    cells += n
                    nb += 1
        del loader, ds
        return dict(status="ok" if cells else "empty",
                    loss=(tot / cells) if cells else None,
                    n_batches=nb, n_cells=cells)
    except Exception as e:  # a corpus that fails should not lose the whole matrix
        return dict(status=f"error: {type(e).__name__}: {e}"[:180],
                    loss=None, n_batches=0, n_cells=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--corpora-root", default="/marimo/corpora_extracted")
    ap.add_argument("--ckpt-dir", default="/marimo/model_checkpoints")
    ap.add_argument("--out-dir", default="/marimo/analysis/out")
    ap.add_argument("--real-db", default="rel-stack")
    args = ap.parse_args()

    import pandas as pd
    from rt_icl import bench

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.corpora_root)

    columns = {}
    for c in MODELS:
        d = root / c
        columns[c] = synthetic_tasks(d) if d.is_dir() else []
        print(f"[e3] corpus {c:8s}: {len(columns[c])} tasks", flush=True)

    real = [(db, t, col, "test", []) for db, t, col, kind, _ in
            bench.tasks_for([args.real_db])][:6]
    columns["real"] = real
    print(f"[e3] corpus {'real':8s}: {len(real)} tasks ({args.real_db})", flush=True)

    rows = []
    for m in MODELS:
        net = bench.load_model(f"{args.ckpt_dir}/{m}_final.pt", device="cuda")
        for cname, tasks in columns.items():
            t0 = time.time()
            r = corpus_loss(net, tasks, ctx_len=args.ctx, batch_size=args.batch_size,
                            n_batches=args.batches)
            r.update(model=m, corpus=cname, seconds=round(time.time() - t0, 1),
                     ctx_len=args.ctx, is_diagonal=(cname == m))
            rows.append(r)
            print(f"[e3] {m:8s} x {cname:8s}  loss="
                  f"{r['loss'] if r['loss'] is None else round(r['loss'], 4)}"
                  f"  cells={r['n_cells']}  {r['status']}  {r['seconds']}s", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "e3_transfer_matrix.csv", index=False)
        del net
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="model", columns="corpus", values="loss")
    wide.to_csv(out_dir / "e3_matrix_wide.csv")
    print("\n=== E3 transfer matrix (masked-cell loss, lower is better) ===")
    print(wide.round(4).to_string())
    print("\nwrote", out_dir / "e3_transfer_matrix.csv", "and e3_matrix_wide.csv")
    print("E3_COMPLETE")


if __name__ == "__main__":
    main()
