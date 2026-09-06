"""
Shortlist items 4 and 7, done properly.

ITEM 4 -- separate the two cross-table routes.
The RT can move information across an FK edge two ways, and both are built from the same
`f2p_nbr_idxs` tensor, which is why shuffling that input corrupts both at once and cannot
tell them apart:

    feat mask = (same_node | kv_in_f2p) & pad     a cell reads its PARENT cells
    nbr  mask = q_in_f2p & pad                    a cell aggregates its CHILDREN

To separate them we intervene on each route alone:
    nbr_zeroed          zero the nbr module's output   -> aggregation removed, parents intact
    feat_parent_blocked patch the feat mask to `same_node & pad`
                        -> parent reading removed, aggregation intact
    both                both at once
Comparing these against `fk_shuffled` (which corrupts both) says which route each arm
actually depends on, rather than inferring it from the architecture.

ITEM 7 -- head knockout, behaviourally.
The weight-space version found head specialization far above the random-init floor but
diffuse in absolute terms. This tests the functional consequence: zero one nbr head's
contribution and measure the loss it costs. Removal is exact rather than approximate --
the block computes `wo @ concat(heads)`, so zeroing `wo`'s columns for head h deletes
exactly that head's contribution and nothing else. Head h is knocked out in all 12 blocks
at once, so this asks "is head index h load-bearing", not "is one layer's head".

Usage:  python analysis/e45_paths_and_heads.py [--part paths|heads|both]
Writes: <out>/e4_path_split.csv, <out>/e7_head_knockout.csv
"""
import argparse
import time
from pathlib import Path

MODELS = ["reldiff", "grdm", "plurel", "rdbpfn"]
N_HEADS, HEAD_DIM = 8, 32


def install_feat_patch(block_parent: bool):
    """Monkey-patch the mask builder so `feat` drops (or keeps) its parent-cell term."""
    import rt.model as rtm
    if not hasattr(rtm, "_ORIG_MASK_MODS"):
        rtm._ORIG_MASK_MODS = rtm._rt_icl_mask_mods

    if not block_parent:
        rtm._rt_icl_mask_mods = rtm._ORIG_MASK_MODS
        return

    def patched(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
        mods = rtm._ORIG_MASK_MODS(node_idxs, f2p_nbr_idxs, col_name_idxs,
                                   table_name_idxs, is_padding)

        def feat_same_node_only(b, h, q, kv):
            same_node = node_idxs[b, q] == node_idxs[b, kv]
            return same_node & (~is_padding[b, q]) & (~is_padding[b, kv])

        mods["feat"] = feat_same_node_only
        return mods

    rtm._rt_icl_mask_mods = patched


class ZeroHead:
    """Temporarily delete one nbr head by zeroing its output-projection columns."""

    def __init__(self, net, head):
        self.net, self.head, self.saved = net, head, []

    def __enter__(self):
        import torch
        lo, hi = self.head * HEAD_DIM, (self.head + 1) * HEAD_DIM
        with torch.no_grad():
            for blk in self.net.blocks:
                w = blk.attns["nbr"].wo.weight
                self.saved.append(w[:, lo:hi].clone())
                w[:, lo:hi] = 0
        return self

    def __exit__(self, *exc):
        import torch
        lo, hi = self.head * HEAD_DIM, (self.head + 1) * HEAD_DIM
        with torch.no_grad():
            for blk, s in zip(self.net.blocks, self.saved):
                blk.attns["nbr"].wo.weight[:, lo:hi] = s
        self.saved = []
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["paths", "heads", "both"])
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ckpt-dir", default="/marimo/model_checkpoints")
    ap.add_argument("--out-dir", default="/marimo/analysis/out")
    ap.add_argument("--tasks", default="user-engagement,user-badge")
    args = ap.parse_args()

    import pandas as pd, torch
    from rt_icl import bench
    from rt_icl.train import PAPER_ARCH
    from rt.model import RelationalTransformer
    from e7_reliance_dose import masked_loss

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    want = set(args.tasks.split(","))
    tasks = [(db, t, c, "test", []) for db, t, c, k, _ in bench.tasks_for(["rel-stack"])
             if t in want]
    kw = dict(ctx_len=args.ctx, batch_size=args.batch_size, n_batches=args.batches)

    def random_init():
        torch.manual_seed(0); a = PAPER_ARCH
        return RelationalTransformer(num_blocks=a["num_blocks"], d_model=a["d_model"],
                                     d_text=a["d_text"], num_heads=a["num_heads"],
                                     d_ff=a["d_ff"]).to("cuda", dtype=torch.bfloat16).eval()

    # ---------------------------------------------------------------- item 4
    if args.part in ("paths", "both"):
        rows = []
        arms = MODELS + ["random_init"]
        for name in arms:
            for cond in ["intact", "nbr_zeroed", "feat_parent_blocked", "both", "fk_shuffled"]:
                install_feat_patch(cond in ("feat_parent_blocked", "both"))
                # the model must be built AFTER the patch: masks are made at forward time,
                # but rebuilding keeps each condition independent of any cached state
                net = (random_init() if name == "random_init"
                       else bench.load_model(f"{args.ckpt_dir}/{name}_final.pt", device="cuda"))
                inner = ("nbr_zeroed" if cond in ("nbr_zeroed", "both")
                         else ("fk_shuffled" if cond == "fk_shuffled" else "intact"))
                r = masked_loss(net, tasks, condition=inner, dose=1.0, **kw)
                r.update(model=name, route_condition=cond)
                rows.append(r)
                print(f"[paths] {name:12s} {cond:20s} loss="
                      f"{r['loss'] if r['loss'] is None else round(r['loss'],4)} "
                      f"{r['status']}", flush=True)
                pd.DataFrame(rows).to_csv(out / "e4_path_split.csv", index=False)
                del net; torch.cuda.empty_cache()
        install_feat_patch(False)
        df = pd.DataFrame(rows)
        piv = df.pivot_table(index="model", columns="route_condition", values="loss")
        if "intact" in piv:
            for c in piv.columns:
                if c != "intact":
                    piv[f"d_{c}"] = piv[c] - piv["intact"]
        print("\n=== item 4: which cross-table route does each arm use? ===")
        print(piv.round(4).to_string())

    # ---------------------------------------------------------------- item 7
    if args.part in ("heads", "both"):
        hrows = []
        for name in MODELS:
            net = bench.load_model(f"{args.ckpt_dir}/{name}_final.pt", device="cuda")
            base = masked_loss(net, tasks, condition="intact", **kw)
            hrows.append(dict(model=name, head=-1, loss=base["loss"], delta=0.0,
                              status=base["status"]))
            print(f"[heads] {name:12s} intact  loss={round(base['loss'],4)}", flush=True)
            for h in range(N_HEADS):
                with ZeroHead(net, h):
                    r = masked_loss(net, tasks, condition="intact", **kw)
                d = (r["loss"] - base["loss"]) if (r["loss"] and base["loss"]) else None
                hrows.append(dict(model=name, head=h, loss=r["loss"], delta=d,
                                  status=r["status"]))
                print(f"[heads] {name:12s} head {h}  loss="
                      f"{r['loss'] if r['loss'] is None else round(r['loss'],4)}"
                      f"  delta={d if d is None else round(d,4)}", flush=True)
                pd.DataFrame(hrows).to_csv(out / "e7_head_knockout.csv", index=False)
            del net; torch.cuda.empty_cache()
        hd = pd.DataFrame(hrows)
        piv = hd[hd.head >= 0].pivot_table(index="model", columns="head", values="delta")
        print("\n=== item 7: loss cost of removing each nbr head (all 12 blocks) ===")
        print(piv.round(4).to_string())
        print("\nconcentration: top head / sum of all heads")
        for m in MODELS:
            v = piv.loc[m].values
            if v.sum() > 0:
                print(f"  {m:9s} top1 share {v.max()/v.sum():.3f}  "
                      f"(uniform = {1/N_HEADS:.3f})  max/min {v.max()/max(v.min(),1e-9):.1f}")
    print("\nPATHS_HEADS_COMPLETE")


if __name__ == "__main__":
    main()
