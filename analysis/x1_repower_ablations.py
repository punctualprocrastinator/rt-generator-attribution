"""
X1 re-power: the benchmark under mechanistic ablation, at honest sample size.

The original X1 table scored the 30k-context ablations at n=256. Re-scoring the
unablated condition at n=2048 moved rel-stack user-engagement from 0.929 to 0.865, so
differences measured in that regime are not trustworthy. This reruns all three
conditions at n=2048.

Two interventions, both matching how the RT actually routes cross-table information:

  fk_shuffled  permute `f2p_nbr_idxs` along the sequence axis. Both cross-table routes
               are derived from this one tensor -- `q_in_f2p` builds the `nbr` mask and
               `kv_in_f2p` builds the parent-cell term of the `feat` mask -- so this
               corrupts the relational graph wherever the model could read it, while
               every cell value and the neighbour-count distribution are untouched.

  nbr_zeroed   zero the output of each block's `nbr` attention. The block computes
               `x = x + attns[a](norms[a](x), ...)` per stream, so returning zeros from
               that module removes precisely the aggregation stream and leaves the
               `col`, `feat` and FFN paths intact.

Batches are drawn with a fixed seed and `shuffle_py(0)`, and the shuffle permutation is
seeded per batch index, so all four models see identical inputs under identical
corruption: the comparison is paired.

Usage:  python analysis/x1_repower_ablations.py [--ctx 30000] [--max-samples 2048]
Writes: <out>/x1_repower.csv
"""
import argparse
import time
from pathlib import Path

MODELS = ["reldiff", "grdm", "plurel", "rdbpfn"]
CONDITIONS = ["intact", "fk_shuffled", "nbr_zeroed"]


def score_one(net, db, task, target, kind, *, condition, ctx_len, batch_size,
              max_samples=2048, split="test", num_workers=2, device="cuda", seed=0):
    import torch
    from sklearn.metrics import roc_auc_score
    from rt.data import RelationalDataset

    rec = dict(db=db, task=task, target=target, kind=kind, condition=condition,
               ctx_len=ctx_len, batch_size=batch_size, split=split)
    hooks = []
    loader = None
    t0 = time.time()
    try:
        if condition == "nbr_zeroed":
            for blk in net.blocks:
                hooks.append(blk.attns["nbr"].register_forward_hook(
                    lambda m, inp, out: torch.zeros_like(out)))

        ds = RelationalDataset(
            tasks=[(db, task, target, split, [])], batch_size=batch_size,
            rank=0, world_size=1, ctx_len=ctx_len, max_bfs_width=128,
            embedding_model="all-MiniLM-L12-v2", d_text=384, seed=seed,
        )
        ds.sampler.shuffle_py(0)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers,
            pin_memory=(device == "cuda"), in_order=True,
        )
        preds, labels, n_seen = [], [], 0
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if n_seen >= max_samples:
                    break
                tbs = batch.pop("true_batch_size")
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                batch["masks"][tbs:, :] = False
                batch["is_targets"][tbs:, :] = False
                batch["is_padding"][tbs:, :] = True

                if condition == "fk_shuffled":
                    f = batch["f2p_nbr_idxs"]
                    g = torch.Generator(device="cpu").manual_seed(1000 + i)
                    perm = torch.randperm(f.shape[1], generator=g).to(f.device)
                    batch["f2p_nbr_idxs"] = f[:, perm, :].contiguous()

                _, yhat = net(batch)
                is_t = batch["is_targets"].bool()
                p = torch.sigmoid(yhat["boolean"][is_t].float()).flatten()
                y = batch["boolean_values"][is_t].float().flatten()
                preds.append(p.cpu()); labels.append(y.cpu())
                n_seen += int(p.numel())

        if not preds:
            rec.update(status="no-batches", n=0)
        else:
            p = torch.cat(preds).numpy(); y = torch.cat(labels).numpy()
            yb = (y > 0).astype(int)
            rec["n"] = int(len(y))
            if yb.min() == yb.max():
                rec.update(status="single-class", auroc=float("nan"))
            else:
                rec.update(status="ok", auroc=float(roc_auc_score(yb, p)),
                           pos_rate=float(yb.mean()), n_pos=int(yb.sum()))
    except torch.cuda.OutOfMemoryError:
        rec.update(status="OOM", n=0)
    except Exception as e:  # one bad cell must not lose the sweep
        rec.update(status=f"{type(e).__name__}: {str(e)[:150]}", n=0)
    finally:
        for h in hooks:
            h.remove()
        del loader
        try:
            import torch as _t; _t.cuda.empty_cache()
        except Exception:
            pass
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=30000)
    ap.add_argument("--max-samples", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--ckpt-dir", default="/marimo/model_checkpoints")
    ap.add_argument("--out-dir", default="/marimo/analysis/out")
    args = ap.parse_args()

    import pandas as pd
    from rt_icl import bench

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in bench.tasks_for(["rel-stack"])
             if t[1] in ("user-engagement", "user-badge")]
    print(f"[x1] {len(MODELS)} models x {len(tasks)} tasks x {len(CONDITIONS)} conditions "
          f"= {len(MODELS)*len(tasks)*len(CONDITIONS)} cells at ctx {args.ctx}", flush=True)

    rows = []
    for m in MODELS:
        net = bench.load_model(f"{args.ckpt_dir}/{m}_final.pt", device="cuda")
        for db, task, target, kind, _ in tasks:
            for cond in CONDITIONS:
                r = score_one(net, db, task, target, kind, condition=cond,
                              ctx_len=args.ctx, batch_size=args.batch_size,
                              max_samples=args.max_samples)
                r["generator"] = m
                rows.append(r)
                a = r.get("auroc")
                print(f"[x1] {m:8s} {task:16s} {cond:12s} "
                      f"auroc={'%.4f' % a if a == a and a is not None else r['status']} "
                      f"n={r.get('n')} npos={r.get('n_pos')} {r['seconds']}s", flush=True)
                pd.DataFrame(rows).to_csv(out / "x1_repower.csv", index=False)
        del net

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["task", "generator"], columns="condition", values="auroc")
    piv = piv[[c for c in CONDITIONS if c in piv.columns]]
    print("\n=== X1 re-powered (AUROC) ===")
    print(piv.round(4).to_string())
    print("\nwrote", out / "x1_repower.csv")
    print("X1_COMPLETE")


if __name__ == "__main__":
    main()
