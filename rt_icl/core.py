"""Canonical intermediate representation + RT-compatible export.

Every generator (RDB-PFN, GRDM, RelDiff, PluRel) is first adapted into the SAME intermediate
representation (`RDBSpec`), then put through the SAME `canonicalize()` and the SAME
`discover_autocomplete_tasks()`. That is what makes the four pretraining arms comparable: any
difference in the trained model has to come from the generated data, not from how we happened to
convert it.

The export target is a RelBench `Database` saved to `<relbench cache>/<db_name>/db/*.parquet`,
which is exactly what RT's Rust preprocessor (`rustler ... pre <db_name>`) consumes.

Hard requirements imposed by the RT preprocessor (`rustler/src/pre.rs`), all enforced here:

  1. FK values are used as DIRECT ROW OFFSETS into the parent table
     (`pnode_idx = parent_offset + fk_value`), so every parent's primary key MUST equal its row
     position, i.e. 0..N-1 in row order. We reindex and remap FKs to guarantee this.
  2. FK columns must be Int64 (`let AnyValue::Int64(val) = val else { panic!() }`); nulls are
     skipped (the row simply loses that edge). We use pandas' nullable "Int64".
  3. Cell dtypes must be one of Boolean / {UInt32,Int16,Int32,Int64,Float32,Float64} / Datetime(ns)
     / String -- anything else hits a `panic!()`. We coerce or drop.
  4. Numeric cells are z-scored per column; a resulting non-finite value triggers `panic!()`.
     We replace +-inf with NaN (skipped downstream) before export.
  5. Boolean columns are z-scored with NO zero-std guard (unlike numeric), so a CONSTANT boolean
     column yields 0/0 = NaN cells that would silently poison the loss. We drop constant booleans.
  6. PK and FK columns are never emitted as cells -- they only define identity and graph edges.
     So "feature columns" (the maskable ones, per the paper's autocomplete definition) are exactly
     the non-PK, non-FK columns.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Semantic types RT can mask and therefore train on (RT paper Sec 3.3: "we only mask cells in
# boolean or numeric columns"). Text/datetime cells are context only.
CLF_KINDS = ("boolean",)
REG_KINDS = ("number",)


@dataclass
class TableSpec:
    """One table: the frame plus the RelBench key/time metadata."""

    df: pd.DataFrame
    pkey_col: str | None = None
    fkey_col_to_pkey_table: dict[str, str] = field(default_factory=dict)
    time_col: str | None = None

    def feature_cols(self) -> list[str]:
        """Columns that become cells (everything that is not a key)."""
        keys = set(self.fkey_col_to_pkey_table) | ({self.pkey_col} if self.pkey_col else set())
        return [c for c in self.df.columns if c not in keys]


RDBSpec = dict[str, TableSpec]


# --------------------------------------------------------------------------------------- #
# canonicalization
# --------------------------------------------------------------------------------------- #
def _is_boolish(s: pd.Series) -> bool:
    """True for a genuine 2-valued column that should be modeled as boolean (-> clf head)."""
    if pd.api.types.is_bool_dtype(s):
        return True
    nun = s.dropna().unique()
    if len(nun) != 2:
        return False
    # {0,1} / {False,True} / {"0","1"} style pairs only -- an arbitrary 2-valued float column
    # stays numeric.
    try:
        vals = sorted(float(v) for v in nun)
    except (TypeError, ValueError):
        return False
    return vals == [0.0, 1.0]


def infer_kind(s: pd.Series, *, max_int_categories: int = 16) -> str:
    """Fallback column kind when an adapter did not declare one."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        nun = s.dropna().nunique()
        if nun <= 1:
            return "numeric"
        # integer-coded low-cardinality columns are nominal codes, not magnitudes
        if pd.api.types.is_integer_dtype(s) and nun <= max_int_categories:
            return "boolean" if nun == 2 else "categorical"
        if _is_boolish(s):
            return "boolean"
        return "numeric"
    return "categorical"


def canonicalize(
    rdb: RDBSpec,
    *,
    max_text_nunique: int = 512,
    max_int_categories: int = 16,
    kinds: dict[str, dict[str, str]] | None = None,
    report: dict | None = None,
) -> RDBSpec:
    """Coerce an adapted RDB into something the RT preprocessor accepts, in place-ish.

    `kinds` maps `table -> column -> kind` with kind in
    {"boolean", "numeric", "categorical", "datetime"}. Adapters fill this in from each generator's
    OWN schema metadata, which is what keeps the semantics right: a 7-level nominal code must
    become a *text* cell (context only), not a numeric one -- otherwise it would be picked up as a
    regression target and the model would be trained to predict a category id with Huber loss.
    Undeclared columns fall back to `infer_kind`.

    Returns the cleaned RDB. `report` (if given) is filled with counts of everything dropped, so
    conversion problems are visible instead of silent.
    """
    rep = report if report is not None else {}
    rep.setdefault("dropped_tables", [])
    rep.setdefault("dropped_cols", [])
    rep.setdefault("dropped_fks", [])
    rep.setdefault("nulled_fk_values", 0)
    rep.setdefault("reindexed_pkeys", 0)
    rep.setdefault("kind_counts", {})
    kinds = kinds or {}

    # ---- 1. drop empty tables, then drop FKs pointing at tables that no longer exist ----
    for name in list(rdb):
        if len(rdb[name].df) == 0:
            rep["dropped_tables"].append(f"{name} (0 rows)")
            del rdb[name]
    for name, t in rdb.items():
        for fk, parent in list(t.fkey_col_to_pkey_table.items()):
            if parent not in rdb or fk not in t.df.columns:
                rep["dropped_fks"].append(f"{name}.{fk} -> {parent} (missing)")
                t.fkey_col_to_pkey_table.pop(fk)
            elif rdb[parent].pkey_col is None:
                # FK offsets are only meaningful against a keyed parent.
                rep["dropped_fks"].append(f"{name}.{fk} -> {parent} (parent has no pkey)")
                t.fkey_col_to_pkey_table.pop(fk)

    # ---- 2. reindex primary keys to 0..N-1 (row order) and build the old->new maps ----
    pk_maps: dict[str, dict] = {}
    for name, t in rdb.items():
        if t.pkey_col is None:
            continue
        t.df = t.df.reset_index(drop=True)
        old = t.df[t.pkey_col]
        new = np.arange(len(t.df), dtype="int64")
        already = pd.api.types.is_integer_dtype(old) and np.array_equal(
            old.to_numpy(dtype="int64", na_value=-1), new
        )
        pk_maps[name] = {} if already else dict(zip(old.tolist(), new.tolist()))
        if not already:
            rep["reindexed_pkeys"] += 1
        t.df[t.pkey_col] = new

    # ---- 3. remap FK values onto the new parent row offsets ----
    for name, t in rdb.items():
        for fk, parent in t.fkey_col_to_pkey_table.items():
            col = t.df[fk]
            pmap = pk_maps.get(parent, {})
            if pmap:
                col = col.map(pmap)
            col = pd.to_numeric(col, errors="coerce")
            n_parent = len(rdb[parent].df)
            bad = col.isna() | (col < 0) | (col >= n_parent)
            if bad.any():
                rep["nulled_fk_values"] += int(bad.sum())
                col = col.where(~bad)
            # nullable Int64 -> parquet int64 with nulls (pre.rs requires Int64, skips nulls)
            t.df[fk] = col.astype("Int64")

    # ---- 4. coerce / drop feature columns according to their declared kind ----
    for name, t in list(rdb.items()):
        keys = set(t.fkey_col_to_pkey_table) | ({t.pkey_col} if t.pkey_col else set())
        tkinds = kinds.get(name, {})
        for c in list(t.df.columns):
            if c in keys:
                continue
            s = t.df[c]
            # unnamed / index leftovers from CSV round-trips
            if str(c).startswith("Unnamed") or str(c) in ("index", "level_0"):
                rep["dropped_cols"].append(f"{name}.{c} (index artifact)")
                t.df = t.df.drop(columns=[c])
                continue
            if s.isna().all():
                rep["dropped_cols"].append(f"{name}.{c} (all null)")
                t.df = t.df.drop(columns=[c])
                continue

            kind = tkinds.get(c) or infer_kind(s, max_int_categories=max_int_categories)
            if c == t.time_col:
                kind = "datetime"

            # A single-valued column carries no information but still consumes one cell of the
            # 1024-cell context window in every sampled sequence. Drop it whatever its kind --
            # this also removes GRDM's deliberate `__placeholder__` constants, and it is the same
            # condition that makes a boolean column produce NaN cells in pre.rs (std = 0).
            if s.dropna().nunique() < 2 and kind != "datetime":
                rep["dropped_cols"].append(f"{name}.{c} (constant {kind})")
                t.df = t.df.drop(columns=[c])
                continue
            rep["kind_counts"][kind] = rep["kind_counts"].get(kind, 0) + 1

            if kind == "datetime":
                dt = pd.to_datetime(s, errors="coerce")
                if getattr(dt.dtype, "tz", None) is not None:
                    dt = dt.dt.tz_localize(None)
                # pre.rs asserts nanosecond datetimes
                t.df[c] = dt.astype("datetime64[ns]")
                if t.df[c].isna().all():
                    rep["dropped_cols"].append(f"{name}.{c} (unparseable datetime)")
                    t.df = t.df.drop(columns=[c])
                    if t.time_col == c:
                        t.time_col = None
                continue

            if kind == "boolean":
                try:
                    b = s.map(lambda v: bool(float(v)) if pd.notna(v) else None)
                except (TypeError, ValueError):
                    codes, _ = pd.factorize(s.astype("object").where(s.notna(), None))
                    b = pd.Series(codes, index=s.index).map(
                        lambda v: None if v < 0 else bool(v)
                    )
                if b.dropna().nunique() < 2:
                    # constant boolean -> std 0 -> NaN cells in pre.rs (no zero-std guard there)
                    rep["dropped_cols"].append(f"{name}.{c} (constant boolean)")
                    t.df = t.df.drop(columns=[c])
                    continue
                t.df[c] = b.astype("boolean")
                continue

            if kind == "numeric":
                f = pd.to_numeric(s, errors="coerce").astype("float64")
                f = f.replace([np.inf, -np.inf], np.nan)  # pre.rs panics on non-finite
                if f.isna().all():
                    rep["dropped_cols"].append(f"{name}.{c} (no finite values)")
                    t.df = t.df.drop(columns=[c])
                    continue
                t.df[c] = f
                continue

            # categorical -> text cell (context only; RT never masks text)
            txt = s.astype("object").where(s.notna(), None)
            nun = pd.Series(txt).dropna().nunique()
            if nun > max_text_nunique:
                rep["dropped_cols"].append(f"{name}.{c} (text nunique={nun})")
                t.df = t.df.drop(columns=[c])
                continue
            t.df[c] = pd.Series(txt).map(
                lambda v: None if v is None else (
                    f"category {int(v)}" if isinstance(v, (int, np.integer)) else str(v)
                )
            ).astype("object")

    # ---- 5. drop tables that became structurally useless ----
    for name in list(rdb):
        t = rdb[name]
        if len(t.df) == 0:
            rep["dropped_tables"].append(f"{name} (0 rows post-clean)")
            del rdb[name]
    # re-drop FKs to tables removed in step 5
    for name, t in rdb.items():
        for fk, parent in list(t.fkey_col_to_pkey_table.items()):
            if parent not in rdb:
                rep["dropped_fks"].append(f"{name}.{fk} -> {parent} (parent dropped)")
                t.fkey_col_to_pkey_table.pop(fk)
                t.df = t.df.drop(columns=[fk], errors="ignore")

    return rdb


# RT represents a row's foreign->primary neighbours in a fixed-width slot: `MAX_F2P_NBRS = 5` in
# rustler/src/fly.rs, and `f2p_nbr_idxs` is shaped (batch, seq_len, 5). A row with more than five
# FK edges trips `assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS)` inside a DataLoader worker,
# which surfaces only as "worker exited unexpectedly". PluRel guards against this upstream too
# (`rt.tasks.is_valid_db` rejects any database with a >5-FK table), so we apply the same rule.
MAX_FKEYS = 5


def validate(rdb: RDBSpec, max_fkeys: int = MAX_FKEYS) -> None:
    """Assert every invariant the RT preprocessor depends on. Raises on the first violation."""
    assert rdb, "empty RDB (no tables)"
    for name, t in rdb.items():
        df = t.df
        assert len(df) > 0, f"{name}: 0 rows"
        n_fk = len(t.fkey_col_to_pkey_table)
        assert n_fk <= max_fkeys, (
            f"{name}: {n_fk} foreign keys > MAX_F2P_NBRS={max_fkeys}. RT stores a row's F->P "
            f"neighbours in a fixed 5-wide slot, so this database cannot be represented"
        )
        if t.pkey_col is not None:
            pk = df[t.pkey_col].to_numpy()
            assert np.array_equal(pk, np.arange(len(df))), (
                f"{name}.{t.pkey_col}: primary key is not 0..N-1 in row order -- FK offsets "
                f"in pre.rs would resolve to the wrong parent rows"
            )
        for fk, parent in t.fkey_col_to_pkey_table.items():
            assert parent in rdb, f"{name}.{fk} -> unknown table {parent}"
            assert str(df[fk].dtype) == "Int64", (
                f"{name}.{fk}: dtype {df[fk].dtype}, expected nullable Int64 (pre.rs panics "
                f"on non-Int64 foreign keys)"
            )
            v = df[fk].dropna()
            if len(v):
                assert v.min() >= 0 and v.max() < len(rdb[parent].df), (
                    f"{name}.{fk}: values out of range for parent {parent} "
                    f"([{v.min()}, {v.max()}] vs {len(rdb[parent].df)} rows)"
                )
        for c in t.feature_cols():
            s = df[c]
            d = str(s.dtype)
            if d == "boolean" or pd.api.types.is_bool_dtype(s):
                assert s.dropna().nunique() >= 2, f"{name}.{c}: constant boolean -> NaN cells"
            elif pd.api.types.is_numeric_dtype(s):
                assert np.isfinite(s.dropna().to_numpy(dtype="float64")).all(), (
                    f"{name}.{c}: non-finite values -> pre.rs panic"
                )
            elif pd.api.types.is_datetime64_any_dtype(s):
                assert d == "datetime64[ns]", f"{name}.{c}: datetime unit {d}, need ns"
            else:
                assert d == "object", f"{name}.{c}: unsupported dtype {d}"


# --------------------------------------------------------------------------------------- #
# autocomplete task discovery  (RT paper Sec 2.2 / App. B)
# --------------------------------------------------------------------------------------- #
def discover_autocomplete_tasks(
    rdb: RDBSpec,
    db_name: str,
    *,
    min_non_missing: int = 64,
    min_minority_frac: float = 0.05,
    tasks_per_db: int | None = 4,
    seed: int = 0,
) -> list[tuple[str, str, str, list[str]]]:
    """Return RT task tuples `(db_name, table_name, target_column, columns_to_drop)`.

    An autocomplete task is masked-cell prediction on a feature column that already exists in the
    database -- no task table is needed (RT paper App. B). We accept a column as a target when it
    carries enough signal to be learnable:

      * boolean  -> binary classification; needs both classes with the minority class at least
                    `min_minority_frac` (a 99.9%-one-class column teaches the model nothing but
                    the prior, and makes the loss look deceptively good).
      * numeric  -> regression; needs non-zero variance.

    `columns_to_drop` is empty: these are synthetic databases with no hand-identified leakage
    columns, and inter-column dependence is exactly the SCM signal we want the model to learn.
    """
    rng = random.Random(seed)
    clf, reg = [], []
    for tname in sorted(rdb):
        t = rdb[tname]
        for c in t.feature_cols():
            if c == t.time_col:
                continue
            s = t.df[c]
            nn = int(s.notna().sum())
            if nn < min_non_missing:
                continue
            if str(s.dtype) == "boolean" or pd.api.types.is_bool_dtype(s):
                vc = s.dropna().astype(bool).value_counts()
                if len(vc) < 2:
                    continue
                if vc.min() / nn < min_minority_frac:
                    continue
                clf.append((db_name, tname, c, []))
            elif pd.api.types.is_numeric_dtype(s):
                v = s.dropna().to_numpy(dtype="float64")
                if v.std(ddof=1) <= 0 or not math.isfinite(float(v.std(ddof=1))):
                    continue
                reg.append((db_name, tname, c, []))

    rng.shuffle(clf)
    rng.shuffle(reg)
    if tasks_per_db is None:
        return clf + reg
    # keep the clf/reg mix as even as the database allows
    half = tasks_per_db // 2
    take_clf = clf[:half]
    take_reg = reg[: tasks_per_db - len(take_clf)]
    if len(take_clf) + len(take_reg) < tasks_per_db:
        take_clf = clf[: tasks_per_db - len(take_reg)]
    return take_clf + take_reg


# --------------------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------------------- #
def relbench_dir(db_name: str, home: str | None = None) -> Path:
    """`$HOME/scratch/relbench/<db_name>` -- what the Rust preprocessor reads."""
    import os

    h = home or os.environ["HOME"]
    return Path(h) / "scratch" / "relbench" / db_name


def save_as_relbench(
    rdb: RDBSpec, db_name: str, home: str | None = None, source: str | None = None
) -> Path:
    """Write the RDB as a RelBench `Database` (parquet + key/time metadata in parquet metadata).

    `source` records which generator database this export came from. Names are positional, so the
    marker is what lets later stages tell "already done" from "done, but for a different database
    that happened to hold this index last time".
    """
    from relbench.base import Database, Table

    table_dict = {
        name: Table(
            df=t.df,
            fkey_col_to_pkey_table=dict(t.fkey_col_to_pkey_table),
            pkey_col=t.pkey_col,
            time_col=t.time_col,
        )
        for name, t in rdb.items()
    }
    out = relbench_dir(db_name, home) / "db"
    out.mkdir(parents=True, exist_ok=True)
    Database(table_dict).save(out)
    if source is not None:
        (relbench_dir(db_name, home) / "rt_icl_source.json").write_text(
            json.dumps({"source": source, "db_name": db_name})
        )
    return out


def verify_saved(db_name: str, home: str | None = None) -> dict:
    """Read the exported parquet back and re-check the pre.rs-critical properties on disk.

    Catches round-trip surprises (notably pyarrow silently writing microsecond timestamps, which
    would trip the `assert unit == Nanoseconds` in pre.rs).
    """
    import pyarrow.parquet as pq

    d = relbench_dir(db_name, home) / "db"
    files = sorted(d.glob("*.parquet"))
    assert files, f"no parquet written for {db_name}"
    info = {"tables": len(files), "rows": 0, "cells": 0}
    for f in files:
        pf = pq.ParquetFile(f)
        meta = pf.schema_arrow.metadata or {}
        for key in (b"pkey_col", b"fkey_col_to_pkey_table", b"time_col"):
            assert key in meta, f"{f.name}: parquet metadata missing {key.decode()}"
        fkeys = json.loads(meta[b"fkey_col_to_pkey_table"])
        pkey = json.loads(meta[b"pkey_col"])
        n = pf.metadata.num_rows
        info["rows"] += n
        for fld in pf.schema_arrow:
            if str(fld.type).startswith("timestamp"):
                assert "ns" in str(fld.type), (
                    f"{f.name}.{fld.name}: {fld.type} -- pre.rs asserts nanosecond datetimes"
                )
            if fld.name in fkeys:
                assert str(fld.type) == "int64", (
                    f"{f.name}.{fld.name}: FK type {fld.type}, pre.rs requires int64"
                )
        keys = set(fkeys) | ({pkey} if pkey else set())
        info["cells"] += n * len([f_ for f_ in pf.schema_arrow.names if f_ not in keys])
    return info
