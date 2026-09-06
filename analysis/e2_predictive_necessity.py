"""
Experiment 2: predictive necessity in the pretraining corpora.

The paper's organizing principle is that a generator teaches cross-table
computation only if its corpus makes a masked cell hard to recover from its
own row. That property is asserted from each generator's design and never
measured. This measures it directly, with no RT involved.

For each target column we fit LightGBM three times:
    A  own-row features only
    B  own row + parent columns joined over FK   (the "read your parents" path)
    C  own row + child aggregates over FK        (the "aggregate your children" path)

parent_lift = R2(B) - R2(A)   child_lift = R2(C) - R2(A)

Splitting the lift this way mirrors the RT's two cross-table routes, so the
data-side measurement lines up with the feat/kv_in_f2p and nbr streams.

Usage:  python analysis/e2_predictive_necessity.py [--dbs N] [--quick]
Writes: analysis/out/e2_results.json, analysis/out/e2_per_task.csv
"""
import argparse
import io
import json
import os
import re
import warnings
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import train_test_split

OUT_DIR = os.path.join("analysis", "out")
SEED = 0
MAX_ROWS = 4000          # rows per table fed to the model
MIN_ROWS = 150           # skip tables smaller than this
MAX_TARGETS_PER_DB = 3

CORPORA = {
    "plurel":  dict(zip_path="data/plurel_databases.zip", fmt="parquet"),
    "rdbpfn":  dict(zip_path="data/rdbpfn_upload.zip", fmt="parquet"),
    "reldiff": dict(zip_path=None, fmt="csv"),   # five per-reference zips
    "grdm":    dict(zip_path="data/synthetic-20260812T162654Z-1-001.zip", fmt="csv"),
}
RELDIFF_ZIPS = [f"data/reldiff/reldiff_rel-{r}.zip" for r in
                ["f1", "avito", "hm", "trial", "event"]]


# ---------------------------------------------------------------- discovery
def list_dbs(corpus, limit):
    """Return [(label, zip_path, [member paths])] for up to `limit` databases."""
    out = []
    if corpus == "reldiff":
        for zp in RELDIFF_ZIPS:
            z = zipfile.ZipFile(zp)
            g = defaultdict(list)
            for m in z.namelist():
                if m.endswith(".csv") and "/RelDiff/" in m:
                    g["/".join(m.split("/")[:-1])].append(m)
            for k in sorted(g)[: max(1, limit // len(RELDIFF_ZIPS))]:
                out.append((k, zp, g[k]))
        return out[:limit]

    zp = CORPORA[corpus]["zip_path"]
    z = zipfile.ZipFile(zp)
    g = defaultdict(list)
    for m in z.namelist():
        if m.endswith((".csv", ".parquet")):
            g["/".join(m.split("/")[:-1])].append(m)
    return [(k, zp, g[k]) for k in sorted(g)[:limit]]


def read_db(zp, members):
    """Return (tables, meta). `meta` is the parsed schema when the corpus ships one."""
    z = zipfile.ZipFile(zp)
    tables, meta = {}, None
    for m in members:
        name = os.path.basename(m).rsplit(".", 1)[0]
        raw = z.read(m)
        if m.endswith((".yaml", ".yml")):
            try:
                import yaml
                meta = yaml.safe_load(raw)
            except Exception:
                pass
            continue
        try:
            df = (pd.read_parquet(io.BytesIO(raw)) if m.endswith(".parquet")
                  else pd.read_csv(io.BytesIO(raw)))
        except Exception:
            continue
        if len(df) > 0:
            tables[name] = df
    return tables, meta


def _contains(child_vals, parent_keys):
    v = child_vals.dropna()
    return float(v.isin(parent_keys).mean()) if len(v) else 0.0


def detect_keys(tables, meta=None):
    """Return (pk_of_table, fk_edges) where fk_edges = [(child, col, parent)].

    Discovery is name- or metadata-driven and value containment is only used to
    verify, because pure containment produces false positives: small-integer
    feature columns (a 4-category code, a race `year`, a finishing `position`)
    are trivially contained in another table's id range and were being picked
    up as foreign keys.
    """
    # ---- RDB-PFN ships an explicit schema; trust it.
    if meta:
        pk, edges = {}, []
        for t in meta.get("tables", []):
            tname = t.get("name")
            for c in t.get("columns", []):
                if c.get("dtype") == "primary_key":
                    pk[tname] = c["name"]
                elif c.get("dtype") == "foreign_key" and c.get("link_to"):
                    parent = str(c["link_to"]).split(".")[0]
                    edges.append((tname, c["name"], parent))
        pk = {t: p for t, p in pk.items() if t in tables}
        edges = [e for e in edges if e[0] in tables and e[2] in tables]
        if pk:
            return pk, edges

    # ---- primary keys: unique integer column, preferring id-shaped names
    pk = {}
    for t, df in tables.items():
        cands = [c for c in df.columns
                 if pd.api.types.is_integer_dtype(df[c]) and df[c].is_unique
                 and df[c].notna().all()]
        if not cands:
            continue
        exact = [c for c in cands if c.lower() in (f"{t.lower()}_id", "row_idx")]
        stem = re.sub(r"[^a-z]", "", t.lower())[:6]
        named = [c for c in cands if re.sub(r"[^a-z]", "", c.lower()).startswith(stem)]
        idish = [c for c in cands if c.lower().endswith(("id", "_id", "idx"))]
        pk[t] = (exact or named or idish or cands)[0]

    pkeys = {t: set(tables[t][c]) for t, c in pk.items()}
    edges = []
    for t, df in tables.items():
        for c in df.columns:
            if c == pk.get(t) or not pd.api.types.is_integer_dtype(df[c]):
                continue

            # PluRel: the name says "foreign key" but not to which table, so the
            # parent is resolved by containment plus key-space coverage. A true
            # parent has most of its keys referenced; a coincidental container
            # (a large table whose id range happens to cover a category code)
            # does not.
            if c.lower().startswith("foreign_row"):
                best, best_cov = None, 0.0
                for p, pcol in pk.items():
                    if p == t or _contains(df[c], pkeys[p]) < 0.98:
                        continue
                    cov = df[c].nunique() / max(len(tables[p]), 1)
                    if cov > best_cov:
                        best, best_cov = p, cov
                if best and best_cov >= 0.10:
                    edges.append((t, c, best))
                continue

            # GRDM / RelDiff: an FK column carries the parent's primary-key name
            # (UserInfo_id, raceId). Require the name match, then verify values.
            for p, pcol in pk.items():
                if p == t:
                    continue
                if c == pcol or c.lower() == f"{p.lower()}_id":
                    if _contains(df[c], pkeys[p]) >= 0.90:
                        edges.append((t, c, p))
                    break
    return pk, edges


# ---------------------------------------------------------------- features
def encode(df):
    """Numeric matrix from a frame: numerics as-is, low-card objects as codes."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[c] = s.astype("float64")
        elif s.dtype == object:
            if s.nunique(dropna=True) <= 200:
                out[c] = s.astype("category").cat.codes.astype("float64")
    return out


def fit_r2(X, y, seed=SEED):
    if X.shape[1] == 0 or len(X) < MIN_ROWS:
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06, num_leaves=31,
                          min_child_samples=20, verbose=-1, random_state=seed,
                          force_col_wise=True)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - np.mean(yte)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def type_census(tables):
    c = defaultdict(int)
    for df in tables.values():
        for col, dt in df.dtypes.items():
            if pd.api.types.is_bool_dtype(dt):
                c["boolean"] += 1
            elif pd.api.types.is_numeric_dtype(dt):
                c["numeric"] += 1
            elif dt == object:
                c["text_or_categorical"] += 1
            else:
                c["other"] += 1
    return dict(c)


def run_db(tables, rng, meta=None):
    """Yield one record per (table, target column) evaluated in this database."""
    pk, edges = detect_keys(tables, meta)
    parents = defaultdict(list)   # child -> [(col, parent)]
    children = defaultdict(list)  # parent -> [(child, col)]
    for ch, col, pa in edges:
        parents[ch].append((col, pa))
        children[pa].append((ch, col))

    keycols = {t: {pk.get(t)} | {c for (cc, c, p) in
                                 [(e[0], e[1], e[2]) for e in edges] if cc == t}
               for t in tables}

    cands = []
    for t, df in tables.items():
        if len(df) < MIN_ROWS or (t not in parents and t not in children):
            continue
        for c in df.columns:
            if c in keycols.get(t, set()):
                continue
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                if s.notna().sum() >= MIN_ROWS and s.nunique() > 10 and s.std(skipna=True) > 0:
                    cands.append((t, c))
    rng.shuffle(cands)

    done = 0
    for t, target in cands:
        if done >= MAX_TARGETS_PER_DB:
            break
        df = tables[t]
        if len(df) > MAX_ROWS:
            df = df.sample(MAX_ROWS, random_state=SEED)
        y = df[target].astype("float64")
        mask = y.notna()
        if mask.sum() < MIN_ROWS:
            continue
        df, y = df[mask], y[mask]

        own = encode(df.drop(columns=[c for c in ([target] + list(keycols.get(t, set())))
                                      if c in df.columns], errors="ignore"))
        r2_a = fit_r2(own, y)
        if r2_a is None:
            continue

        # ---- parent features
        par = own.copy()
        for col, pa in parents.get(t, []):
            if col not in df.columns:
                continue
            pdf = tables[pa]
            ppk = pk.get(pa)
            if ppk is None:
                continue
            pfeat = encode(pdf.drop(columns=[ppk], errors="ignore"))
            pfeat[ppk] = pdf[ppk].values
            j = df[[col]].merge(pfeat, left_on=col, right_on=ppk, how="left").drop(columns=[ppk])
            j.index = df.index
            par = par.join(j.add_prefix(f"par_{pa}_"))
        r2_b = fit_r2(par, y) if par.shape[1] > own.shape[1] else None

        # ---- child aggregate features
        ch_feats = own.copy()
        tpk = pk.get(t)
        if tpk is not None:
            for cht, ccol in children.get(t, []):
                cdf = tables[cht]
                if ccol not in cdf.columns:
                    continue
                num = cdf.select_dtypes("number").drop(columns=[ccol], errors="ignore")
                agg = num.groupby(cdf[ccol]).agg(["mean", "std", "count"])
                agg.columns = [f"chi_{cht}_{a}_{b}" for a, b in agg.columns]
                cnt = cdf.groupby(ccol).size().rename(f"chi_{cht}_n")
                agg = agg.join(cnt, how="outer")
                j = df[[tpk]].merge(agg, left_on=tpk, right_index=True, how="left")
                j.index = df.index
                ch_feats = ch_feats.join(j.drop(columns=[tpk]))
        r2_c = fit_r2(ch_feats, y) if ch_feats.shape[1] > own.shape[1] else None

        done += 1
        yield dict(table=t, target=target, n=int(len(df)),
                   n_own=int(own.shape[1]),
                   n_par=int(par.shape[1] - own.shape[1]),
                   n_chi=int(ch_feats.shape[1] - own.shape[1]),
                   r2_own=r2_a, r2_parent=r2_b, r2_child=r2_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", type=int, default=25)
    ap.add_argument("--corpora", default="plurel,rdbpfn,reldiff,grdm")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows, census = [], {}
    for corpus in args.corpora.split(","):
        dbs = list_dbs(corpus, args.dbs)
        print(f"\n=== {corpus}: {len(dbs)} databases")
        ccount = defaultdict(int)
        for i, (label, zp, members) in enumerate(dbs):
            tables, meta = read_db(zp, members)
            if len(tables) < 2:
                continue
            for k, v in type_census(tables).items():
                ccount[k] += v
            try:
                for rec in run_db(tables, rng, meta):
                    rec.update(corpus=corpus, db=label)
                    rows.append(rec)
            except Exception as e:
                print(f"   [{label}] skipped: {type(e).__name__}: {e}")
            if (i + 1) % 5 == 0:
                print(f"   {i+1}/{len(dbs)} dbs, {len(rows)} tasks")
        census[corpus] = dict(ccount)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/e2_per_task.csv", index=False)

    summary = {}
    print("\n" + "=" * 78)
    print(f"{'corpus':9s} {'tasks':>6s} {'R2 own':>8s} {'parent lift':>12s} {'child lift':>11s}")
    for corpus, g in df.groupby("corpus"):
        pl = (g.r2_parent - g.r2_own).dropna()
        cl = (g.r2_child - g.r2_own).dropna()
        summary[corpus] = dict(
            n_tasks=int(len(g)), r2_own=float(g.r2_own.mean()),
            parent_lift_mean=float(pl.mean()) if len(pl) else None,
            parent_lift_median=float(pl.median()) if len(pl) else None,
            parent_lift_se=float(pl.std() / np.sqrt(len(pl))) if len(pl) > 1 else None,
            n_parent=int(len(pl)),
            child_lift_mean=float(cl.mean()) if len(cl) else None,
            child_lift_median=float(cl.median()) if len(cl) else None,
            child_lift_se=float(cl.std() / np.sqrt(len(cl))) if len(cl) > 1 else None,
            n_child=int(len(cl)),
            type_census=census.get(corpus, {}),
        )
        print(f"{corpus:9s} {len(g):6d} {g.r2_own.mean():8.3f} "
              f"{pl.mean():+12.4f} {cl.mean():+11.4f}")

    with open(f"{OUT_DIR}/e2_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ntype census: {json.dumps(census, indent=2)}")
    print(f"wrote {OUT_DIR}/e2_results.json and e2_per_task.csv")


if __name__ == "__main__":
    main()
