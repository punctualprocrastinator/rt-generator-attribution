"""One adapter per generator: native output -> `RDBSpec`.

Each adapter does ONLY format translation (find the tables, find the keys, label each column's
kind). All cleaning, key reindexing, dtype coercion and task selection happen afterwards in
`core.canonicalize` / `core.discover_autocomplete_tasks`, identically for all four generators --
so the four pretraining arms differ in their DATA, never in their treatment.

Native formats
--------------
RDB-PFN  `dag_rdb_<i>/` : `metadata.yaml` (tables[].columns[].dtype in
         {primary_key, foreign_key, float, category, timestamp}, FKs carry
         `link_to: "<parent>.<parent_pk>"`) + `<table>.parquet`. Categories are dense int codes.

GRDM     (rdb-diffusion) : synthetic tables as `<table>_synthetic.csv` (or `<table>.csv`), schema in
         a separate reference dir: `dataset_meta.json` ({tables: {t: {parents, children}}}) plus
         `<table>_domain.json` ({col: {size, type: "discrete"|"continuous"}}). PK is `<t>_id`,
         FK to parent p is `<p>_id`. Note: rdb-diffusion has no datetime columns -- the reference
         builder converts timestamps to numeric days -- so GRDM databases are non-temporal.

RelDiff  (SyntheRela/SDV) : tables as `<table>.csv`, schema in `metadata.json`
         ({tables: {t: {columns: {c: {sdtype}}, primary_key}}, relationships: [...]}).

PluRel   : native RelBench `Database` straight from `plurel.SyntheticDataset` -- no files needed.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from glob import glob
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import yaml

from .core import RDBSpec, TableSpec


# --------------------------------------------------------------------------------------- #
# RDB-PFN
# --------------------------------------------------------------------------------------- #
def from_rdbpfn(rdb_dir: str | Path) -> tuple[RDBSpec, dict[str, dict[str, str]]]:
    """Read one `dag_rdb_<i>` directory. Returns (rdb, kinds)."""
    rdb_dir = Path(rdb_dir)
    meta = yaml.safe_load((rdb_dir / "metadata.yaml").read_text(encoding="utf-8"))
    rdb: RDBSpec = {}
    kinds: dict[str, dict[str, str]] = {}

    for t in meta.get("tables") or []:
        tname = t["name"]
        pq = rdb_dir / f"{tname}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        # a repeated FK column name (same parent twice) would collide -- keep the first
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        pkey, fkeys, tkinds = None, {}, {}
        for c in t.get("columns") or []:
            cname, dtype = c["name"], c.get("dtype")
            if cname not in df.columns:
                continue
            if dtype == "primary_key":
                pkey = cname
            elif dtype == "foreign_key":
                link = c.get("link_to") or ""
                parent = link.split(".")[0] if "." in link else None
                if parent:
                    fkeys[cname] = parent
            elif dtype == "float":
                tkinds[cname] = "numeric"
            elif dtype == "category":
                n = int(c.get("num_categories") or 0)
                # 2 levels -> boolean (a real binary-classification target); more levels are
                # NOMINAL codes and must not be modeled as magnitudes
                tkinds[cname] = "boolean" if n == 2 else "categorical"
            elif dtype == "timestamp":
                tkinds[cname] = "datetime"

        time_col = t.get("time_column") or None
        if time_col and time_col not in df.columns:
            time_col = None
        rdb[tname] = TableSpec(df, pkey, fkeys, time_col)
        kinds[tname] = tkinds

    return rdb, kinds


def iter_rdbpfn(root: str | Path, limit: int | None = None) -> Iterator[tuple[str, RDBSpec, dict]]:
    """Yield (source_id, rdb, kinds) for every `dag_rdb_*` under `root` (recursively)."""
    root = Path(root)
    dirs = sorted(
        (p for p in root.rglob("dag_rdb_*") if p.is_dir() and (p / "metadata.yaml").exists()),
        key=lambda p: (p.parent.name, int(p.name.rsplit("_", 1)[1])),
    )
    for i, d in enumerate(dirs):
        if limit is not None and i >= limit:
            return
        rdb, kinds = from_rdbpfn(d)
        yield f"{d.parent.name}/{d.name}", rdb, kinds


# --------------------------------------------------------------------------------------- #
# GRDM  (rdb-diffusion)
# --------------------------------------------------------------------------------------- #
def _find_grdm_schema(sample_dir: Path, schema_dir: str | Path | None) -> Path:
    """Locate the dir holding `dataset_meta.json` for a GRDM sample."""
    if schema_dir is not None:
        p = Path(schema_dir)
        if (p / "dataset_meta.json").exists():
            return p
        raise FileNotFoundError(f"no dataset_meta.json in schema_dir {p}")
    for cand in (sample_dir, *sample_dir.parents[:4]):
        if (cand / "dataset_meta.json").exists():
            return cand
        for sib in sorted(cand.glob("*_ref35")) + sorted(cand.glob("*sampled_structure*")):
            if (sib / "dataset_meta.json").exists():
                return sib
    raise FileNotFoundError(
        f"could not locate dataset_meta.json for {sample_dir}; pass schema_dir explicitly "
        f"(the rdb-diffusion reference dir, e.g. data/<db>_ref35)"
    )


def from_grdm_inferred(sample_dir: str | Path) -> tuple[RDBSpec, dict[str, dict[str, str]]]:
    """Read a GRDM sample WITHOUT `dataset_meta.json`, inferring the schema from column names.

    rdb-diffusion's own format is self-describing on the key columns: a table `t` stores its primary
    key as `t_id` and a foreign key to parent `p` as `p_id`. Column KINDS cannot be recovered
    exactly (the `*_domain.json` discrete/continuous flag is what says whether an integer column is
    a nominal code), so those are left undeclared and `core.infer_kind` decides. Prefer passing a
    real `schema_dir` when you have one -- this is the fallback for synthetic samples that were
    persisted without their reference schema.
    """
    sample_dir = Path(sample_dir)
    frames: dict[str, pd.DataFrame] = {}
    for csv in sorted(sample_dir.glob("*.csv")):
        tname = csv.stem
        if tname.endswith("_synthetic"):
            tname = tname[: -len("_synthetic")]
        frames[tname] = pd.read_csv(csv)

    rdb: RDBSpec = {}
    for tname, df in frames.items():
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        pkey = f"{tname}_id" if f"{tname}_id" in df.columns else None
        fkeys = {
            f"{p}_id": p
            for p in frames
            if p != tname and f"{p}_id" in df.columns
        }
        rdb[tname] = TableSpec(df, pkey, fkeys, None)
    return rdb, {t: {} for t in rdb}


def reconstruct_grdm_schema(
    sample_dirs: list[str | Path],
    out_dir: str | Path,
    *,
    max_categories: int = 1000,
    verbose: bool = True,
) -> dict:
    """Rebuild `dataset_meta.json` + `<table>_domain.json` from GRDM samples themselves.

    The reference schema normally lives on the ephemeral Colab disk of the sampling run, so it is
    usually gone by the time we pretrain. It can be recovered from the samples because
    rdb-diffusion's own converter guarantees two things:

      * keys are named by convention -- table `t` has primary key `t_id`, and a foreign key to
        parent `p` is `p_id`;
      * every DISCRETE column is written as **dense integer codes 0..K-1** (the converter asserts
        exactly this: "categorical codes not dense"), while continuous columns are floats.

    So a column whose observed values are precisely {0, 1, ..., K-1} is discrete with `size = K`,
    and anything else is continuous. That is far more faithful than guessing from dtype alone: a
    50-level category would otherwise be read as a numeric column and picked as a *regression*
    target.

    Pass several samples of the SAME base database so every category level is observed; a level
    missing from one small sample would otherwise break the density test.
    """
    import numpy as np

    frames: dict[str, list[pd.DataFrame]] = {}
    for d in sample_dirs:
        for csv in sorted(Path(d).glob("*.csv")):
            t = csv.stem[: -len("_synthetic")] if csv.stem.endswith("_synthetic") else csv.stem
            frames.setdefault(t, []).append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(f"no CSVs found under {sample_dirs}")

    tables = {t: pd.concat(v, ignore_index=True) for t, v in frames.items()}
    parents_of = {
        t: sorted(p for p in tables if p != t and f"{p}_id" in df.columns)
        for t, df in tables.items()
    }
    children_of: dict[str, list[str]] = {t: [] for t in tables}
    for t, ps in parents_of.items():
        for p in ps:
            children_of[p].append(t)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, str]] = {}

    for t, df in tables.items():
        keys = {f"{t}_id", *(f"{p}_id" for p in parents_of[t])}
        domain, kinds = {}, {}
        for c in df.columns:
            if c in keys:
                continue
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            discrete = False
            size = int(s.nunique()) if len(s) else 0
            if len(s):
                integral = bool(np.allclose(s.to_numpy(), np.round(s.to_numpy())))
                u = np.unique(np.round(s.to_numpy()).astype("int64")) if integral else None
                # dense codes 0..K-1 -> a categorical, whatever K is
                discrete = bool(
                    integral
                    and u is not None
                    and u.min() == 0
                    and u.max() < max_categories
                    and len(u) == u.max() + 1
                )
                if discrete:
                    size = int(u.max()) + 1
            domain[c] = {"size": size, "type": "discrete" if discrete else "continuous"}
            kinds[c] = "discrete" if discrete else "continuous"
        (out_dir / f"{t}_domain.json").write_text(json.dumps(domain))
        summary[t] = kinds

    # Kahn order: parents before children, matching to_rdbdiff_dataset's relation_order
    order, done, remaining = [], set(), dict(parents_of)
    while remaining:
        ready = sorted(t for t, ps in remaining.items() if all(p in done for p in ps))
        if not ready:
            raise RuntimeError(f"cyclic schema among {sorted(remaining)}")
        for t in ready:
            order.extend([[p, t] for p in parents_of[t]] if parents_of[t] else [[None, t]])
            done.add(t)
            remaining.pop(t)

    meta = {
        "relation_order": order,
        "tables": {
            t: {"children": sorted(children_of[t]), "parents": parents_of[t]} for t in tables
        },
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta))

    if verbose:
        print(f"[grdm-schema] {out_dir}")
        for t in sorted(tables):
            n_disc = sum(1 for v in summary[t].values() if v == "discrete")
            n_cont = len(summary[t]) - n_disc
            print(
                f"    {t:22s} rows={len(tables[t]):>7,} parents={parents_of[t] or '-'} "
                f"discrete={n_disc} continuous={n_cont}"
            )
    return {"tables": summary, "meta": meta, "out_dir": str(out_dir)}


def reconstruct_grdm_schemas_under(root: str | Path, verbose: bool = True) -> list[str]:
    """Reconstruct a schema for every group of GRDM samples sharing a parent directory.

    Writes `dataset_meta.json` + `<table>_domain.json` into each parent, which is exactly where
    `from_grdm`'s auto-discovery looks -- so afterwards `SCHEMA_DIR = None` resolves the correct
    schema per base database, even when the tree holds several of them.
    """
    root = Path(root)
    groups: dict[Path, list[Path]] = {}
    for csv in sorted(root.rglob("*.csv")):
        d = csv.parent
        if (d / "dataset_meta.json").exists():
            continue
        groups.setdefault(d.parent, []).append(d)
    written = []
    for parent, dirs in sorted(groups.items()):
        if (parent / "dataset_meta.json").exists():
            # a real reference schema is already there -- never overwrite it with a reconstruction
            if verbose:
                print(f"[grdm-schema] {parent.name}: real schema present, leaving it alone")
            continue
        uniq = sorted(set(dirs))
        if verbose:
            print(f"\n[grdm-schema] {parent.name}: {len(uniq)} sample dirs")
        reconstruct_grdm_schema(uniq[:8], parent, verbose=verbose)
        written.append(str(parent))
    return written


def from_grdm(
    sample_dir: str | Path,
    schema_dir: str | Path | None = None,
    allow_inferred_schema: bool = False,
) -> tuple[RDBSpec, dict[str, dict[str, str]]]:
    """Read one GRDM synthetic database directory (CSVs) using an external schema dir."""
    sample_dir = Path(sample_dir)
    try:
        sdir = _find_grdm_schema(sample_dir, schema_dir)
    except FileNotFoundError:
        if not allow_inferred_schema:
            raise
        return from_grdm_inferred(sample_dir)
    meta = json.loads((sdir / "dataset_meta.json").read_text())
    tables_meta = meta["tables"]

    rdb: RDBSpec = {}
    kinds: dict[str, dict[str, str]] = {}
    for tname, tmeta in tables_meta.items():
        csv = sample_dir / f"{tname}_synthetic.csv"
        if not csv.exists():
            csv = sample_dir / f"{tname}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        pkey = f"{tname}_id" if f"{tname}_id" in df.columns else None
        fkeys = {
            f"{p}_id": p
            for p in tmeta.get("parents", [])
            if f"{p}_id" in df.columns and p in tables_meta
        }

        tkinds = {}
        dom_path = sdir / f"{tname}_domain.json"
        if dom_path.exists():
            for col, info in json.loads(dom_path.read_text()).items():
                if col not in df.columns:
                    continue
                if info.get("type") == "discrete":
                    size = int(info.get("size") or 0)
                    tkinds[col] = "boolean" if size == 2 else "categorical"
                else:
                    tkinds[col] = "numeric"
        rdb[tname] = TableSpec(df, pkey, fkeys, None)  # rdb-diffusion data is non-temporal
        kinds[tname] = tkinds

    return rdb, kinds


def iter_grdm(
    root: str | Path,
    schema_dir: str | Path | None = None,
    limit: int | None = None,
    allow_inferred_schema: bool = False,
) -> Iterator[tuple[str, RDBSpec, dict]]:
    """Yield every GRDM sample dir under `root` (a dir containing table CSVs)."""
    root = Path(root)
    seen, dirs, refs = set(), [], []
    for csv in sorted(root.rglob("*.csv")):
        d = csv.parent
        if d in seen:
            continue
        seen.add(d)
        # A reference dir (`<db>_ref35`, or a `sampled_structure`) also holds CSVs -- but those are
        # REAL RelBench rows, not generated samples. Sweeping them in would put real data into the
        # generator's pretraining corpus and quietly contaminate this arm of the study. They are
        # distinguishable because only reference dirs carry the schema next to the tables.
        if (d / "dataset_meta.json").exists() or d.name.endswith("_ref35"):
            refs.append(d)
            continue
        dirs.append(d)
    if refs:
        print(
            f"[grdm] ignoring {len(refs)} reference/schema dir(s) holding real rows, e.g. "
            f"{refs[0].name} -- only generated samples are used as training data"
        )
    for i, d in enumerate(sorted(dirs)):
        if limit is not None and i >= limit:
            return
        try:
            rdb, kinds = from_grdm(d, schema_dir, allow_inferred_schema=allow_inferred_schema)
        except FileNotFoundError:
            continue
        if rdb:
            yield str(d.relative_to(root)), rdb, kinds


# --------------------------------------------------------------------------------------- #
# RelDiff  (SyntheRela / SDV metadata)
# --------------------------------------------------------------------------------------- #
def _load_sdv_metadata(path: Path) -> dict:
    m = json.loads(path.read_text())
    if "tables" not in m and len(m) == 1:  # some SDV versions nest under a single key
        m = next(iter(m.values()))
    return m


def from_reldiff(
    sample_dir: str | Path,
    metadata_path: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> tuple[RDBSpec, dict[str, dict[str, str]]]:
    """Read one RelDiff sample dir (CSVs) plus its SDV/SyntheRela `metadata.json`.

    The NEAREST `metadata.json` at or above `sample_dir` wins, so one tree can hold several base
    databases and each sample still resolves against its own schema.

    The walk stops at `root` rather than after a fixed number of levels. The archived layout nests
    samples as `<family>/synthetic/<base>/RelDiff_gen_sub<k>/sample<i>`, which puts the family root
    exactly four parents up -- right at the edge of the previous fixed bound, so any extra wrapper
    directory (a hand-made zip with its own top-level folder is the obvious way to get one) would
    have silently pushed the schema out of reach and skipped every sample in that family.
    """
    sample_dir = Path(sample_dir)
    if metadata_path is None:
        stop = Path(root).resolve() if root is not None else None
        for cand in (sample_dir, *sample_dir.parents):
            if (cand / "metadata.json").exists():
                metadata_path = cand / "metadata.json"
                break
            if stop is not None and cand.resolve() == stop:
                break
    if metadata_path is None:
        raise FileNotFoundError(f"no metadata.json found for {sample_dir}")
    meta = _load_sdv_metadata(Path(metadata_path))

    # child -> {fk_col: parent}
    fk_of: dict[str, dict[str, str]] = {}
    for rel in meta.get("relationships") or []:
        fk_of.setdefault(rel["child_table_name"], {})[rel["child_foreign_key"]] = rel[
            "parent_table_name"
        ]

    rdb: RDBSpec = {}
    kinds: dict[str, dict[str, str]] = {}
    for tname, tmeta in (meta.get("tables") or {}).items():
        csv = sample_dir / f"{tname}.csv"
        if not csv.exists():
            continue
        cols_meta = tmeta.get("columns") or {}
        parse_dates = [c for c, i in cols_meta.items() if i.get("sdtype") == "datetime"]
        df = pd.read_csv(csv)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        pkey = tmeta.get("primary_key")
        if pkey is not None and pkey not in df.columns:
            pkey = None
        fkeys = {c: p for c, p in fk_of.get(tname, {}).items() if c in df.columns}

        tkinds = {}
        for c, info in cols_meta.items():
            if c not in df.columns or c == pkey or c in fkeys:
                continue
            sdtype = info.get("sdtype")
            if sdtype == "boolean":
                tkinds[c] = "boolean"
            elif sdtype == "numerical":
                tkinds[c] = "numeric"
            elif sdtype == "datetime":
                tkinds[c] = "datetime"
            elif sdtype == "categorical":
                tkinds[c] = "boolean" if df[c].dropna().nunique() == 2 else "categorical"

        # the first datetime column is the table's time column
        time_col = next((c for c in parse_dates if c in df.columns), None)
        rdb[tname] = TableSpec(df, pkey, fkeys, time_col)
        kinds[tname] = tkinds

    return rdb, kinds


def extract_zips(
    pattern: str | Path, dest: str | Path, *, verbose: bool = True
) -> Path:
    """Extract every zip matching `pattern` into `dest` (one subfolder per zip). Resumable.

    RelDiff runs are archived one zip per base database (rel-f1.zip, rel-hm.zip, ...) under
    `MyDrive/reldiff_run/`. **Each zip is extracted into its own `dest/<zip stem>/` folder, and that
    isolation is load-bearing**: the sampling notebooks build the archive with
    `make_archive(..., base_dir="synthetic")`, so every zip has the SAME internal top-level folder
    (`synthetic/`) and extracting them into one directory would have them overwrite each other.
    Keeping them apart also gives each base database a private root to hang its `metadata.json`
    off, which is what per-sample schema discovery in `from_reldiff` resolves against.
    """
    import zipfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zips = sorted(Path(p) for p in glob(str(pattern)))
    if not zips:
        raise FileNotFoundError(f"no zips matched {pattern}")
    for z in zips:
        out = dest / z.stem
        if out.is_dir() and any(out.iterdir()):
            if verbose:
                print(f"[zip] {z.name}: already extracted -> {out}")
            continue
        t0 = time.time()
        if verbose:
            print(f"[zip] extracting {z.name} ({z.stat().st_size/1e9:.2f} GB) ...", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            # __MACOSX/ resource forks shadow real names and would be picked up as sample dirs
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX/")]
            zf.extractall(out, members=members)
        if verbose:
            n = sum(1 for _ in out.rglob("*") if _.is_file())
            tops = sorted({p.name for p in out.iterdir()})
            print(
                f"[zip] {z.name}: {n:,} files in {(time.time()-t0)/60:.1f} min  "
                f"top-level={tops[:4]}"
            )
    return dest


# --------------------------------------------------------------------------------------- #
def _reldiff_family(path: Path, root: Path) -> str:
    """The base-database folder a sample belongs to = its first path part under `root`.

    With one zip per base database that is the zip stem, so samples group by base database even
    though the tree below it (`synthetic/<base>/RelDiff_gen_sub<k>/sample<i>`) is several levels
    deep and repeats the same folder names in every family.
    """
    rel = path.relative_to(root).parts
    return rel[0] if rel else path.name


def reldiff_hoist_metadata(
    root: str | Path, *, extra_dir: str | Path | None = None, verbose: bool = True
) -> dict[str, str | None]:
    """Put each base database's `metadata.json` where per-sample discovery will find it.

    RelDiff's sampler calls `save_tables(syn, .../sample<i>)` **without** `save_metadata=True`
    (see `sample_all_structures.py`), and the sampling notebooks persist only
    `reldiff_run/synthetic/`. So a sample dir is CSV-only, and an archive carries a schema at all
    only if the reference dataset dir -- `<base>_ref35/`, which `save_reference` DOES write with
    `save_metadata=True` -- happened to be zipped alongside the samples.

    Wherever such a `metadata.json` exists under a family, copy it to that family's root, the first
    place `from_reldiff`'s upward walk looks. Families with no schema are reported as `None` rather
    than patched over: RelDiff table and column names carry no key-naming convention (unlike GRDM's
    `<table>_id`), so there is nothing to infer keys from, and guessing would silently invent
    foreign keys -- which `canonicalize` would then turn into confidently wrong parent edges.

    `extra_dir` is the recovery path when the archives turned out not to contain a schema: drop
    `<family>.json` (or `<family>/metadata.json`) there -- next to the zips on Drive is convenient
    -- and it is used for that family. Those are small files, so this avoids re-zipping the corpus.
    """
    root = Path(root)
    extra = Path(extra_dir) if extra_dir else None
    found: dict[str, str | None] = {}
    if verbose:
        # Say up front what the supply folder actually holds. Without this, a missing or misplaced
        # METADATA_DIR is indistinguishable from schemas that failed to match, and both surface
        # only as "schema: MISSING" further down.
        if extra is None:
            print("[reldiff] no METADATA_DIR set; schemas must come from inside the zips")
        elif not extra.is_dir():
            print(f"[reldiff] METADATA_DIR DOES NOT EXIST: {extra}")
        else:
            avail = sorted(p.name for p in extra.glob("*.json"))
            print(
                f"[reldiff] METADATA_DIR {extra}: "
                f"{len(avail)} json file(s){' ' + str(avail[:8]) if avail else ' -- EMPTY'}"
            )
    for fam in sorted(p for p in root.iterdir() if p.is_dir()):
        dst = fam / "metadata.json"
        if dst.exists():
            found[fam.name] = str(dst)
            continue
        supplied = None
        if extra is not None:
            for cand in (extra / f"{fam.name}.json", extra / fam.name / "metadata.json"):
                if cand.exists():
                    supplied = cand
                    break
            if supplied is None:
                # Tolerate the zip stem carrying a prefix the schema file does not (the archives are
                # named `reldiff_rel-f1.zip` while the reference schema is `rel-f1`'s), matching on
                # containment either way. Longest match wins; a tie is ambiguous and must not be
                # resolved by guessing, since the wrong schema would mean the wrong foreign keys.
                hits = sorted(
                    (p for p in extra.glob("*.json") if p.stem in fam.name or fam.name in p.stem),
                    key=lambda p: len(p.stem),
                    reverse=True,
                )
                if len(hits) > 1 and len(hits[0].stem) == len(hits[1].stem):
                    raise ValueError(
                        f"{fam.name}: ambiguous schema in {extra} -- "
                        f"{[h.name for h in hits[:2]]} match equally well. "
                        f"Rename one to exactly {fam.name}.json."
                    )
                if hits:
                    supplied = hits[0]
        if supplied is not None:
            shutil.copy(supplied, dst)
            found[fam.name] = str(dst)
            if verbose:
                print(f"[reldiff] {fam.name}: schema supplied from {supplied}")
            continue
        cands = sorted(fam.rglob("metadata.json"))
        if not cands:
            found[fam.name] = None
            continue
        # prefer the reference dataset the samples were drawn from over a subgraph's cut-down copy:
        # `<base>_ref35` carries the FULL schema, and `from_reldiff` skips tables whose CSV is absent
        # anyway, so the full schema is the safe superset for both full and subgraph samples.
        pick = next((c for c in cands if c.parent.name.endswith("_ref35")), cands[0])
        shutil.copy(pick, dst)
        found[fam.name] = str(dst)
        if verbose:
            print(f"[reldiff] {fam.name}: schema {pick.relative_to(root)} -> {dst.name}")
    return found


def reldiff_source_report(root: str | Path) -> dict:
    """Per-sample-group inventory: how many samples, and whether a schema can be found.

    RelDiff writes tables as CSV with no schema, so `metadata.json` is mandatory -- and unlike
    GRDM there is no `<table>_id` naming convention to fall back on. Checking up front turns a
    silent '0 databases converted' into an actionable message.
    """
    root = Path(root)
    groups: dict[str, dict] = {}
    for d in reldiff_sample_dirs(root):
        key = _reldiff_family(d, root)
        g = groups.setdefault(
            key, {"samples": 0, "metadata": None, "example": str(d), "tables": []}
        )
        g["samples"] += 1
        g["tables"].append(sum(1 for _ in d.glob("*.csv")))
        if g["metadata"] is None:
            for cand in (d, *d.parents):
                if (cand / "metadata.json").exists():
                    g["metadata"] = str(cand / "metadata.json")
                    break
                if cand == root:
                    break
    for g in groups.values():
        t = g.pop("tables")
        g["tables_min"], g["tables_max"] = (min(t), max(t)) if t else (0, 0)
    return groups


def reldiff_sample_dirs(root: str | Path) -> list[Path]:
    """Every `sample*` dir holding CSVs, sorted. Falls back to any CSV-bearing dir."""
    root = Path(root)
    dirs = sorted({p for p in root.rglob("sample*") if p.is_dir() and any(p.glob("*.csv"))})
    if not dirs:  # fall back: any dir with CSVs
        dirs = sorted({p.parent for p in root.rglob("*.csv")})
    return dirs


def _interleave_families(dirs: list[Path], root: Path) -> list[Path]:
    """Round-robin the sample dirs across base databases.

    `convert_all` stops as soon as it has kept TARGET_DBS databases, and the five RelDiff families
    together produce more samples than the study's per-generator budget. Taken in sorted order that
    cap lands *inside one family*: rel-avito and rel-event in full, part of rel-f1, and nothing at
    all from rel-hm or rel-trial. The arm would then be a near-alphabetical slice of the generator
    rather than a sample of it, and would not be comparable with the GRDM arm built from the same
    five base databases. Interleaving makes the truncation fall evenly across all five.
    """
    from itertools import zip_longest

    groups: dict[str, list[Path]] = {}
    for d in dirs:
        groups.setdefault(_reldiff_family(d, root), []).append(d)
    out: list[Path] = []
    for tier in zip_longest(*(groups[k] for k in sorted(groups))):
        out.extend(d for d in tier if d is not None)
    return out


def iter_reldiff(
    root: str | Path,
    metadata_path: str | Path | None = None,
    limit: int | None = None,
    *,
    interleave: bool = True,
) -> Iterator[tuple[str, RDBSpec, dict]]:
    """Yield every RelDiff `sample*` dir under `root`, round-robin across base databases."""
    root = Path(root)
    dirs = reldiff_sample_dirs(root)
    if interleave:
        dirs = _interleave_families(dirs, root)
    for i, d in enumerate(dirs):
        if limit is not None and i >= limit:
            return
        try:
            rdb, kinds = from_reldiff(d, metadata_path, root=root)
        except FileNotFoundError:
            continue
        if rdb:
            yield str(d.relative_to(root)), rdb, kinds


# --------------------------------------------------------------------------------------- #
# PluRel  (native RelBench Database)
# --------------------------------------------------------------------------------------- #
def from_relbench_db(
    db, stypes: dict[str, dict[str, str]] | None = None
) -> tuple[RDBSpec, dict[str, dict[str, str]]]:
    """Wrap a RelBench `Database` as an RDBSpec.

    `stypes` maps `table -> column -> torch_frame stype name` ("categorical"/"numerical"/...). When
    given (PluRel exposes it on the dataset) it decides the column kinds; otherwise we fall back to
    dtype inference in `core.canonicalize`.
    """
    rdb: RDBSpec = {}
    kinds: dict[str, dict[str, str]] = {}
    stypes = stypes or {}
    for tname, table in db.table_dict.items():
        df = table.df.reset_index(drop=True)
        pkey = table.pkey_col
        fkeys = dict(table.fkey_col_to_pkey_table)
        tstypes = stypes.get(tname, {})
        tkinds = {}
        for c in df.columns:
            if c == pkey or c in fkeys:
                continue
            if c == table.time_col:
                tkinds[c] = "datetime"
                continue
            st = tstypes.get(c)
            if st == "categorical":
                tkinds[c] = "boolean" if df[c].dropna().nunique() == 2 else "categorical"
            elif st == "numerical":
                tkinds[c] = "numeric"
            elif st in ("timestamp", "datetime"):
                tkinds[c] = "datetime"
            # else: leave undeclared -> canonicalize infers
        rdb[tname] = TableSpec(df, pkey, fkeys, table.time_col)
        kinds[tname] = tkinds
    return rdb, kinds


def _plurel_stypes(ds) -> dict[str, dict[str, str]]:
    """Pull per-column stypes out of a PluRel SyntheticDataset's schema graph."""
    out: dict[str, dict[str, str]] = {}
    graph = getattr(ds, "table_relationships", None)
    if graph is None:
        return out
    for node_id in graph.nodes:
        attrs = graph.nodes[node_id]
        tname = attrs.get("name")
        if tname is None:
            continue
        cols = {}
        for col, info in (attrs.get("columns") or {}).items():
            st = info.get("_stype") or info.get("stype")
            if st is not None:
                cols[col] = getattr(st, "value", str(st))
        out[tname] = cols
    return out


_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _plurel_pool_init() -> None:
    """Pin each worker to a single compute thread.

    PluRel's SCM sampling is numpy/torch linear algebra, and by default every worker starts one BLAS
    thread per core. With P workers on C cores that is P*C threads contending for C cores, which
    turns a ~17 s database into minutes. This repo already hit the same trap generating RDB-PFN
    (see generated/rdbpfn/README.md). Parallelism here comes from the processes, not from BLAS.
    """
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _plurel_generate_one(args) -> tuple[int, str]:
    """Worker: generate one PluRel database and cache it. Module-level so it is picklable."""
    seed, cache_root = args
    cache = Path(cache_root) / f"plurel-{seed}"
    if (cache / "db").is_dir() and (cache / "stypes.json").exists():
        return seed, "cached"
    try:
        from plurel import Config, SyntheticDataset

        ds = SyntheticDataset(seed=seed, config=Config())
        db = ds.make_db()
        cache.mkdir(parents=True, exist_ok=True)
        db.save(cache / "db")
        # The stypes live on the dataset object, not in the saved Database, so persist them here.
        # Without this a cached reload would fall back to guessing column kinds from dtypes --
        # exactly the mistake that turns a multi-level category into a regression target.
        (cache / "stypes.json").write_text(json.dumps(_plurel_stypes(ds)))
        return seed, "generated"
    except Exception as e:  # noqa: BLE001 - one bad seed must not kill the pool
        return seed, f"FAILED {type(e).__name__}: {e}"


def generate_plurel_corpus(
    num_dbs: int,
    cache_root: str | Path,
    *,
    seed_offset: int = 0,
    num_proc: int | None = None,
    progress_every: int = 25,
) -> dict:
    """Generate `num_dbs` PluRel databases in parallel and cache them.

    Generation is pure CPU work at roughly 17 s per database, so 1000 of them is ~4.7 hours in one
    process. PluRel ships a multiprocessing generator for the same reason; this mirrors it while
    also persisting the per-column stypes the conversion step needs. Resumable: an already-cached
    seed is skipped.
    """
    import multiprocessing as mp

    cache_root = Path(cache_root).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    num_proc = num_proc or max(1, (os.cpu_count() or 2) - 1)
    seeds = [seed_offset + i for i in range(num_dbs)]

    # Children inherit the parent's environment, so cap threads BEFORE the pool is created --
    # by the time a worker runs, numpy/BLAS has already read these.
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    for v in _THREAD_VARS:
        os.environ[v] = "1"

    t0 = time.time()
    print(f"[plurel] generating {num_dbs} databases with {num_proc} processes -> {cache_root}",
          flush=True)
    print(f"[plurel] threads capped to 1 per worker ({num_proc} procs on "
          f"{os.cpu_count()} vCPUs); first completions reported individually", flush=True)
    done, cached, failed = 0, 0, []
    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(processes=num_proc, initializer=_plurel_pool_init) as pool:
            for seed, status in pool.imap_unordered(
                _plurel_generate_one, [(s, str(cache_root)) for s in seeds], chunksize=1
            ):
                done += 1
                if status == "cached":
                    cached += 1
                elif status.startswith("FAILED"):
                    failed.append((seed, status))
                # report the first few individually: silence for 25 completions makes a working
                # pool indistinguishable from a hung one
                if done <= 10 or done % progress_every == 0 or done == len(seeds):
                    el = time.time() - t0
                    rate = done / max(el, 1e-9)
                    eta = (len(seeds) - done) / max(rate, 1e-9)
                    print(f"[plurel] {done}/{len(seeds)}  {el/60:.1f} min  "
                          f"{rate*60:.1f} db/min  ETA {eta/60:.0f} min  failed={len(failed)}",
                          flush=True)
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
    if failed:
        print(f"[plurel] {len(failed)} failed, e.g. {failed[:3]}")
    print(f"[plurel] done in {(time.time()-t0)/60:.1f} min "
          f"({cached} already cached, {len(failed)} failed)")
    return {"num_dbs": num_dbs, "cached": cached, "failed": failed, "cache_root": str(cache_root)}


def iter_plurel(
    num_dbs: int,
    seed_offset: int = 0,
    cache_dir: str | Path | None = None,
    limit: int | None = None,
    preset: str = "main",
) -> Iterator[tuple[str, RDBSpec, dict]]:
    """Yield PluRel databases as RDBSpecs, reading the cache when one exists.

    With `cache_dir` pointing at a `generate_plurel_corpus` run this is a fast disk read and the
    stypes come back exactly as generated. Without a cache it falls back to generating in-process,
    which is correct but slow (~17 s per database).
    """
    from relbench.base import Database

    n = num_dbs if limit is None else min(num_dbs, limit)
    for k in range(n):
        seed = seed_offset + k
        cache = Path(cache_dir).expanduser() / f"plurel-{seed}" if cache_dir else None
        if cache is not None and (cache / "db").is_dir() and (cache / "stypes.json").exists():
            db = Database.load(cache / "db")
            stypes = json.loads((cache / "stypes.json").read_text())
        else:
            from plurel import Config, SyntheticDataset

            ds = SyntheticDataset(seed=seed, config=Config())
            db = ds.make_db()
            stypes = _plurel_stypes(ds)
        rdb, kinds = from_relbench_db(db, stypes)
        yield f"plurel-{seed}", rdb, kinds
