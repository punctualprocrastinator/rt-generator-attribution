"""
E7 re-run + dose-response + random-init floor (shortlist items 2 and 3, and a check
on the paper's Table 3).

Three things at once, because they share one loop:

  * E7 verification. Masked-cell loss under intact / nbr-zeroed / FK-blanked / FK-shuffled
    inputs, the same four conditions the paper reports, so the published table can be
    checked rather than trusted.

  * Dose-response (item 3). FK links are corrupted at 0/25/50/75/100 percent instead of
    only all-or-nothing. Two points cannot show a trend; a monotone curve for one model
    and flat lines for the rest is far harder to explain away as batch noise, because
    noise does not usually arrange itself monotonically in a corruption fraction.

  * Random-init floor (item 2). An untrained RT is scored under identical conditions.
    Without it, "PluRel's loss rises 0.9%" has no scale: we cannot tell indifference from
    a model that never learned to read neighbours at all.

Corruption acts on `f2p_nbr_idxs`, from which both cross-table masks are derived
(`q_in_f2p` builds `nbr`, `kv_in_f2p` builds the parent term of `feat`), so shuffling it
degrades every route the model could use. Zeroing the `nbr` module's output instead
removes only the aggregation stream, leaving the parent path intact -- the two conditions
therefore separate the two routes.

Identical batches across every model and condition (fixed seed, `shuffle_py(0)`).

Usage:  python analysis/e7_reliance_dose.py [--batches 24]
Writes: <out>/e7_reliance.csv, <out>/e7_dose_response.csv
"""
import argparse
import time
from pathlib import Path

MODELS = ["reldiff", "grdm", "plurel", "rdbpfn"]
DOSES = [0.0, 0.25, 0.50, 0.75, 1.0]


def corrupt(batch, condition, dose, i, torch):
    """Return the batch with the requested corruption applied in place."""
    if condition == "fk_blanked":
        batch["f2p_nbr_idxs"] = torch.full_like(batch["f2p_nbr_idxs"], -1)
    elif condition == "fk_shuffled":
        f = batch["f2p_nbr_idxs"]
        S = f.shape[1]
        k = int(round(dose * S))
        if k > 1:
            g = torch.Generator(device="cpu").manual_seed(1000 + i)
            idx = torch.randperm(S, generator=g)[:k]          # which rows to scramble
            perm = idx[torch.randperm(k, generator=g)]        # scrambled amongst themselves
            new = f.clone()
            new[:, idx, :] = f[:, perm, :]
            batch["f2p_nbr_idxs"] = new.contiguous()
    return batch


def masked_loss(net, tasks, *, condition="intact", dose=1.0, ctx_len=1024,
                batch_size=16, n_batches=24, split="test", num_workers=2,
                device="cuda", seed=0):
    import torch
    from rt.data import RelationalDataset

    hooks = []
    rec = dict(condition=condition, dose=dose, ctx_len=ctx_len)
    t0 = time.time()
    try:
        if condition == "nbr_zeroed":
            for blk in net.blocks:
                hooks.append(blk.attns["nbr"].register_forward_hook(
                    lambda m, inp, out: torch.zeros_like(out)))

        ds = RelationalDataset(
            tasks=list(tasks), batch_size=batch_size, rank=0, world_size=1,
            ctx_len=ctx_len, max_bfs_width=128,
            embedding_model="all-MiniLM-L12-v2", d_text=384, seed=seed,
        )
        ds.sampler.shuffle_py(0)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers,
            pin_memory=(device == "cuda"), in_order=True)

        tot, cells, nb = 0.0, 0, 0
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                tbs = batch.pop("true_batch_size")
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                batch["masks"][tbs:, :] = False
                batch["is_targets"][tbs:, :] = False
                batch["is_padding"][tbs:, :] = True
                batch = corrupt(batch, condition, dose, i, torch)
                loss, _ = net(batch)
                n = int(batch["is_targets"].bool().sum())
                if n:
                    tot += float(loss) * n; cells += n; nb += 1
        rec.update(status="ok" if cells else "empty",
                   loss=(tot / cells) if cells else None, n_batches=nb, n_cells=cells)
        del loader, ds
    except Exception as e:
        rec.update(status=f"{type(e).__name__}: {str(e)[:130]}", loss=None)
    finally:
        for h in hooks:
            h.remove()
        try:
            import torch as _t; _t.cuda.empty_cache()
        except Exception:
            pass
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ckpt-dir", default="/marimo/model_checkpoints")
    ap.add_argument("--out-dir", default="/marimo/analysis/out")
    ap.add_argument("--tasks", default="user-engagement,user-badge",
                    help="prepared rel-stack tasks to score")
    args = ap.parse_args()

    import pandas as pd, torch
    from rt_icl import bench
    from rt_icl.train import PAPER_ARCH
    from rt.model import RelationalTransformer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # only the tasks whose archives were actually prepared; asking for an unprepared one
    # raises a bare KeyError from the sampler rather than anything diagnosable
    want = set(args.tasks.split(","))
    tasks = [(db, t, c, "test", []) for db, t, c, k, _ in bench.tasks_for(["rel-stack"])
             if t in want]
    assert tasks, f"none of {sorted(want)} found in rel-stack"
    print(f"[e7] {len(tasks)} rel-stack tasks {[t[1] for t in tasks]}, "
          f"ctx {args.ctx}, {args.batches} batches", flush=True)

    def random_init():
        torch.manual_seed(0)
        a = PAPER_ARCH
        net = RelationalTransformer(num_blocks=a["num_blocks"], d_model=a["d_model"],
                                    d_text=a["d_text"], num_heads=a["num_heads"],
                                    d_ff=a["d_ff"])
        # bench.load_model casts checkpoints to bfloat16 on CUDA; the floor has to match
        # or the batch's bf16 activations meet fp32 weights and every cell errors out.
        return net.to("cuda", dtype=torch.bfloat16).eval()

    arms = [(m, lambda m=m: bench.load_model(f"{args.ckpt_dir}/{m}_final.pt", device="cuda"))
            for m in MODELS] + [("random_init", random_init)]

    rel_rows, dose_rows = [], []
    for name, make in arms:
        net = make()
        # --- the four E7 conditions
        for cond in ["intact", "nbr_zeroed", "fk_shuffled", "fk_blanked"]:
            r = masked_loss(net, tasks, condition=cond, dose=1.0, ctx_len=args.ctx,
                            batch_size=args.batch_size, n_batches=args.batches)
            r["model"] = name; rel_rows.append(r)
            print(f"[e7] {name:12s} {cond:12s} loss="
                  f"{r['loss'] if r['loss'] is None else round(r['loss'],4)} "
                  f"cells={r.get('n_cells')} {r['seconds']}s", flush=True)
            pd.DataFrame(rel_rows).to_csv(out / "e7_reliance.csv", index=False)
        # --- dose-response on FK shuffling
        for d in DOSES:
            r = masked_loss(net, tasks, condition="fk_shuffled", dose=d, ctx_len=args.ctx,
                            batch_size=args.batch_size, n_batches=args.batches)
            r["model"] = name; dose_rows.append(r)
            print(f"[dose] {name:12s} frac={d:.2f} loss="
                  f"{r['loss'] if r['loss'] is None else round(r['loss'],4)}", flush=True)
            pd.DataFrame(dose_rows).to_csv(out / "e7_dose_response.csv", index=False)
        del net
        torch.cuda.empty_cache()

    rel = pd.DataFrame(rel_rows)
    piv = rel.pivot_table(index="model", columns="condition", values="loss")
    if "intact" in piv:
        for c in ["nbr_zeroed", "fk_shuffled", "fk_blanked"]:
            if c in piv:
                piv[f"d_{c}"] = piv[c] - piv["intact"]
    print("\n=== E7: masked-cell loss ===")
    print(piv.round(4).to_string())

    dose = pd.DataFrame(dose_rows).pivot_table(index="model", columns="dose", values="loss")
    print("\n=== Dose-response (FK shuffle fraction) ===")
    print(dose.round(4).to_string())
    print("\nE7_COMPLETE")


if __name__ == "__main__":
    main()
