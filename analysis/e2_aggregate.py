"""
Robust aggregation of the experiment 2 per-task results.

Raw R2 is unbounded below, so a single task where the model diverges (one GRDM
`Price` column reached -28383) swamps a mean over ~30 tasks. An R2 below 0
already means "no better than predicting the mean", and how far below carries
no extra information about neighbour usefulness, so scores are clipped to
[-1, 1] before differencing and the headline statistic is the median with a
bootstrap interval.

Usage:  python analysis/e2_aggregate.py
Writes: analysis/out/e2_summary.json, analysis/out/fig_e2_lift.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join("analysis", "out")
ORDER = ["reldiff", "grdm", "plurel", "rdbpfn"]
SEED = 0


def boot_ci(x, fn=np.median, n=5000, seed=SEED):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    d = pd.read_csv(f"{OUT}/e2_per_task.csv")
    for c in ["r2_own", "r2_parent", "r2_child"]:
        d[c + "_c"] = d[c].clip(-1, 1)
    d["parent_lift"] = d.r2_parent_c - d.r2_own_c
    d["child_lift"] = d.r2_child_c - d.r2_own_c

    summary = {}
    print(f"{'corpus':9s} {'tasks':>5s} {'R2own med':>10s} "
          f"{'parent lift (median [95% CI])':>34s} {'child lift':>26s}")
    for c in ORDER:
        g = d[d.corpus == c]
        if not len(g):
            continue
        pl = g.parent_lift.dropna()
        cl = g.child_lift.dropna()
        pci, cci = boot_ci(pl), boot_ci(cl)
        summary[c] = dict(
            n_tasks=int(len(g)),
            r2_own_median=float(g.r2_own_c.median()),
            parent_lift_median=float(pl.median()) if len(pl) else None,
            parent_lift_ci=pci, n_parent=int(len(pl)),
            parent_lift_frac_positive=float((pl > 0.01).mean()) if len(pl) else None,
            child_lift_median=float(cl.median()) if len(cl) else None,
            child_lift_ci=cci, n_child=int(len(cl)),
            child_lift_frac_positive=float((cl > 0.01).mean()) if len(cl) else None,
        )
        print(f"{c:9s} {len(g):5d} {g.r2_own_c.median():10.3f} "
              f"{pl.median():+10.3f} [{pci[0]:+.3f}, {pci[1]:+.3f}]  "
              f"{cl.median():+8.3f} [{cci[0]:+.3f}, {cci[1]:+.3f}]")

    with open(f"{OUT}/e2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- CSV exports ----------
    srows = []
    for c, s in summary.items():
        srows.append(dict(
            corpus=c, n_tasks=s["n_tasks"], r2_own_median=s["r2_own_median"],
            parent_lift_median=s["parent_lift_median"],
            parent_lift_ci_lo=s["parent_lift_ci"][0], parent_lift_ci_hi=s["parent_lift_ci"][1],
            parent_lift_frac_positive=s["parent_lift_frac_positive"], n_parent=s["n_parent"],
            child_lift_median=s["child_lift_median"],
            child_lift_ci_lo=s["child_lift_ci"][0], child_lift_ci_hi=s["child_lift_ci"][1],
            child_lift_frac_positive=s["child_lift_frac_positive"], n_child=s["n_child"],
        ))
    sdf = pd.DataFrame(srows).set_index("corpus").loc[[c for c in ORDER if c in summary]]
    # descriptive: share of predictable signal contributed by neighbours
    sdf["relational_share"] = [
        (max(r.parent_lift_median, 0) /
         (max(r.r2_own_median, 0) + max(r.parent_lift_median, 0)))
        if (max(r.r2_own_median, 0) + max(r.parent_lift_median, 0)) > 0 else np.nan
        for r in sdf.itertuples()]
    sdf.to_csv(f"{OUT}/e2_summary.csv")

    # per-task table with the clipped columns and lifts used for the headline
    d.to_csv(f"{OUT}/e2_per_task_clipped.csv", index=False)

    # type census, one row per corpus/type
    try:
        with open(f"{OUT}/e2_results.json") as f:
            raw = json.load(f)
        crows = [dict(corpus=c, column_type=k, n_columns=v)
                 for c, s in raw.items() for k, v in s.get("type_census", {}).items()]
        if crows:
            pd.DataFrame(crows).to_csv(f"{OUT}/e2_type_census.csv", index=False)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, key, title in [(axes[0], "parent_lift", "Parent lift (read your parents)"),
                           (axes[1], "child_lift", "Child lift (aggregate your children)")]:
        vals = [d[d.corpus == c][key].dropna().values for c in ORDER]
        # matplotlib renamed boxplot's `labels` to `tick_labels` in 3.9 and removed the
        # old spelling in 3.11, so pick whichever this install accepts.
        import inspect
        lab_kw = ("tick_labels" if "tick_labels"
                  in inspect.signature(ax.boxplot).parameters else "labels")
        ax.boxplot(vals, showfliers=False, medianprops=dict(color="crimson"),
                   **{lab_kw: ORDER})
        for i, v in enumerate(vals):
            ax.scatter(np.random.default_rng(SEED).normal(i + 1, 0.06, len(v)), v,
                       s=9, alpha=0.5, color="steelblue")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("gain in held-out $R^2$ from\nadding cross-table features", fontsize=9)
    fig.suptitle("Neighbour predictive information in each pretraining corpus", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_e2_lift.png", dpi=200)
    print(f"\nwrote {OUT}/e2_summary.json and fig_e2_lift.png")


if __name__ == "__main__":
    main()
