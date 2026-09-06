"""
E6: linear probes on frozen representations.

Ablation asks what a model *uses*; probing asks what it *encodes*. A model can encode
parent identity and still ignore it, so this separates "the corpus never taught it" from
"the objective never required it".

Probes are split deliberately into two kinds, because not all of them are informative:

  sanity   sem_type, table identity. These are fed to the model as inputs and embedded
           directly, so every arm including an untrained one should score near ceiling.
           They are here to prove the extraction and probe pipeline works, not to
           discriminate between models. A low score here means a bug, not a finding.

  relational  row degree (how many FK neighbours the cell's row has) and parent-table
           identity (which table this cell's FK edge points at). Neither is a per-cell
           input: recovering them requires aggregating over the sequence or following an
           edge. These are the probes that can actually separate the arms.

Everything is reported against a random-init RT, so the number is the gain from
pretraining rather than the gain from having a 256-d representation at all.

Usage:  python analysis/e6_linear_probes.py [--batches 16]
Writes: <out>/e6_probes.csv
"""
import argparse
import time
from pathlib import Path

MODELS = ["reldiff", "grdm", "plurel", "rdbpfn"]


def collect(net, tasks, *, layer, ctx_len=1024, batch_size=8, n_batches=16,
            split="test", num_workers=2, device="cuda", seed=0, max_cells=40000):
    """Residual-stream activations after `layer`, plus probe labels, for real cells."""
    import numpy as np
    import torch
    from rt.data import RelationalDataset

    store = {}

    def hook(m, inp, out):
        store["h"] = out.detach()

    h = net.blocks[layer].register_forward_hook(hook)
    X, lab = [], {k: [] for k in ["sem_type", "table", "degree", "parent_table"]}
    try:
        ds = RelationalDataset(tasks=list(tasks), batch_size=batch_size, rank=0,
                               world_size=1, ctx_len=ctx_len, max_bfs_width=128,
                               embedding_model="all-MiniLM-L12-v2", d_text=384, seed=seed)
        ds.sampler.shuffle_py(0)
        loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers,
                                             pin_memory=True, in_order=True)
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                tbs = batch.pop("true_batch_size")
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                batch["is_padding"][tbs:, :] = True
                net(batch)
                rep = store["h"].float()                       # (B, S, d)
                keep = ~batch["is_padding"]                    # real cells only

                f2p = batch["f2p_nbr_idxs"]                    # (B, S, 5), -1 = empty
                # degree: how many FK slots this cell's row actually uses
                degree = (f2p >= 0).sum(-1)

                # parent-table identity: table of the node the first FK slot points at.
                # node_idxs maps position -> node id, so invert it per batch element.
                first = f2p[:, :, 0]                           # (B, S)
                node = batch["node_idxs"]
                tbl = batch["table_name_idxs"]
                B, S = node.shape
                ptab = torch.full_like(first, -1)
                for b in range(B):
                    # position of each node id, last occurrence wins (cells share a node)
                    lookup = torch.full((int(node[b].max()) + 2,), -1, dtype=torch.long,
                                        device=node.device)
                    lookup[node[b].long()] = torch.arange(S, device=node.device)
                    ok = first[b] >= 0
                    idx = first[b].long().clamp(min=0, max=lookup.numel() - 1)
                    pos = lookup[idx]
                    valid = ok & (pos >= 0)
                    ptab[b][valid] = tbl[b][pos[valid]]

                m = keep & (ptab >= 0)                         # parent-table probe subset
                X.append(rep[keep].cpu().numpy())
                lab["sem_type"].append(batch["sem_types"][keep].cpu().numpy())
                lab["table"].append(tbl[keep].cpu().numpy())
                lab["degree"].append(degree[keep].cpu().numpy())
                pt = torch.where(m, ptab, torch.full_like(ptab, -1))
                lab["parent_table"].append(pt[keep].cpu().numpy())
                if sum(x.shape[0] for x in X) >= max_cells:
                    break
        del loader, ds
    finally:
        h.remove()
    import numpy as np
    return np.concatenate(X), {k: np.concatenate(v) for k, v in lab.items()}


def probe(X, y, kind, seed=0):
    """Linear probe with an honest split; returns accuracy (clf) or R2 (reg)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ok = y >= 0
    X, y = X[ok], y[ok]
    if len(y) < 500:
        return None, len(y), None
    if kind == "clf":
        vals, counts = np.unique(y, return_counts=True)
        if len(vals) < 2:
            return None, len(y), None
        majority = counts.max() / counts.sum()          # the baseline that matters
    else:
        if y.std() == 0:
            return None, len(y), None
        majority = None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed,
                                          stratify=y if kind == "clf" else None)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "clf":
        m = LogisticRegression(max_iter=300, n_jobs=-1, multi_class="auto")
        m.fit(Xtr, ytr)
        return float(m.score(Xte, yte)), len(y), float(majority)
    m = Ridge(alpha=1.0).fit(Xtr, ytr)
    return float(m.score(Xte, yte)), len(y), None


PROBES = [("sem_type", "clf", "sanity"), ("table", "clf", "sanity"),
          ("degree", "reg", "relational"), ("parent_table", "clf", "relational")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ckpt-dir", default="/marimo/model_checkpoints")
    ap.add_argument("--out-dir", default="/marimo/analysis/out")
    ap.add_argument("--tasks", default="user-engagement,user-badge")
    args = ap.parse_args()

    import pandas as pd, torch
    from rt_icl import bench
    from rt_icl.train import PAPER_ARCH
    from rt.model import RelationalTransformer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    want = set(args.tasks.split(","))
    tasks = [(db, t, c, "test", []) for db, t, c, k, _ in bench.tasks_for(["rel-stack"])
             if t in want]

    def random_init():
        torch.manual_seed(0); a = PAPER_ARCH
        return RelationalTransformer(num_blocks=a["num_blocks"], d_model=a["d_model"],
                                     d_text=a["d_text"], num_heads=a["num_heads"],
                                     d_ff=a["d_ff"]).to("cuda", dtype=torch.bfloat16).eval()

    arms = [(m, lambda m=m: bench.load_model(f"{args.ckpt_dir}/{m}_final.pt", device="cuda"))
            for m in MODELS] + [("random_init", random_init)]

    rows = []
    for name, make in arms:
        net = make()
        t0 = time.time()
        X, lab = collect(net, tasks, layer=args.layer, batch_size=args.batch_size,
                         n_batches=args.batches)
        print(f"[e6] {name:12s} {X.shape[0]} cells, d={X.shape[1]}, "
              f"{time.time()-t0:.0f}s", flush=True)
        for target, kind, group in PROBES:
            score, n, base = probe(X, lab[target], kind)
            rows.append(dict(model=name, probe=target, kind=kind, group=group,
                             score=score, n=n, majority_baseline=base))
            print(f"[e6]   {target:14s} ({group:10s}) "
                  f"{'acc' if kind=='clf' else 'R2'}="
                  f"{'%.4f' % score if score is not None else 'n/a'}"
                  f"{'' if base is None else f'  (majority {base:.3f})'}", flush=True)
            pd.DataFrame(rows).to_csv(out / "e6_probes.csv", index=False)
        del net, X, lab
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="probe", columns="model", values="score")
    cols = [c for c in MODELS + ["random_init"] if c in piv.columns]
    piv = piv[cols]
    if "random_init" in piv.columns:
        for m in MODELS:
            if m in piv.columns:
                piv[f"d_{m}"] = piv[m] - piv["random_init"]
    print("\n=== E6 probes (gain over random init in d_ columns) ===")
    print(piv.round(4).to_string())
    print("\nE6_COMPLETE")


if __name__ == "__main__":
    main()
