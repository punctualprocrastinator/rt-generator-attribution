"""Build a bounded, referentially-complete subsample of a RelBench database.

rel-amazon does not fit: `Dataset.get_db()` materializes every table as a pandas frame, which
exhausts RAM before anything can be filtered, and `rustler pre` then needs its own full copy. This
module never materializes a full table. It reads the cached parquet with duckdb, filters during the
read, and only the SUBSAMPLE is ever held in memory.

What the sample is
------------------
**Entity-anchored, not a random row sample.** A random sample of rows would cut foreign keys and
`pre` resolves FK values as direct row offsets into the parent, so a dangling key does not raise --
it silently points at the wrong parent row. Instead:

  1. seed on the entities the TASK actually scores (its test split), so every sampled row is one
     that contributes a prediction;
  2. expand DOWN to the rows that reference a seed (this is the relational context RT reads);
  3. expand UP to every parent any kept row references (required for referential integrity);
  4. repeat to a fixpoint, then drop anything still unresolved.

Steps 2/3 are the same two-phase closure `RelDiff_Schemas` uses, and for the same reason: doing
them interleaved can oscillate, while pull-then-drop is monotone and terminates.

What it is NOT
--------------
No temporal cut. The reference-corpus builder deliberately kept only the first 35% of time to avoid
leaking the test period into *training data*; here we are BENCHMARKING, so the test window must
stay intact. Sampling is over entities, never over time.

Comparability
-------------
Subsampling changes how hard a task is, but it changes it **identically for every generator arm**,
because all four are scored on the same prepared database. Within-task comparisons -- which is what
the ranking uses -- stay valid. The database is written under a distinct name so a subsampled
result can never be mistaken for a full-database one.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

#: Parquet key-value metadata keys RelBench writes and `pre` reads back.
PKEY_KEY = "pkey_col"
FKEY_KEY = "fkey_col_to_pkey_table"


def _decode_meta(raw: dict | None) -> dict:
    """Parquet KV metadata comes back as bytes->bytes; decode to str->str."""
    if not raw:
        return {}
    return {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in raw.items()
    }


def read_schema(db_dir: str | Path) -> dict:
    """Discover {table: {pkey, fkeys, time_col, path, n_rows}} without reading any row data."""
    import pyarrow.parquet as pq

    out: dict[str, dict] = {}
    for p in sorted(Path(db_dir).glob("*.parquet")):
        md = pq.read_metadata(p)
        meta = _decode_meta(md.schema.to_arrow_schema().metadata)
        pkey = json.loads(meta.get(PKEY_KEY, "null"))
        fkeys = json.loads(meta.get(FKEY_KEY, "{}")) or {}
        out[p.stem] = {
            "pkey": pkey,
            "fkeys": fkeys,
            "time_col": json.loads(meta.get("time_col", "null")),
            "path": str(p),
            "n_rows": md.num_rows,
            "_raw_meta": meta,
        }
    if not out:
        raise FileNotFoundError(f"no parquet tables under {db_dir}")
    return out


def _write_table(con, sql: str, src_path: str, dst: Path) -> int:
    """Materialize `sql` and write it to `dst`, carrying the source's key-value metadata.

    The schema (`pkey_col`, `fkey_col_to_pkey_table`) lives in that metadata and is the ONLY place
    `pre` reads keys from -- a subsampled table written without it parses as a keyless table, and
    every relational edge silently disappears.
    """
    import pyarrow.parquet as pq

    res = con.execute(sql)
    # duckdb >=1.5 `.arrow()` hands back a streaming RecordBatchReader, older versions a Table
    tbl = res.fetch_arrow_table() if hasattr(res, "fetch_arrow_table") else res.arrow()
    if hasattr(tbl, "read_all"):
        tbl = tbl.read_all()
    src_meta = _decode_meta(pq.read_schema(src_path).metadata)
    keep = {k: v for k, v in src_meta.items() if not k.startswith("pandas")}
    merged = dict(_decode_meta(tbl.schema.metadata))
    merged.update(keep)
    tbl = tbl.replace_schema_metadata(merged)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, dst)
    return tbl.num_rows


def subsample_db(
    src_db_dir: str | Path,
    src_tasks_dir: str | Path,
    out_root: str | Path,
    task_names: list[str],
    *,
    n_entities: int = 3000,
    max_children_per_parent: int = 200,
    max_rows_per_table: int = 300_000,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Write a referentially-complete subsample to `out_root/{db,tasks}`.

    `n_entities` is per task and applies to the TEST split -- those are the rows that become
    predictions, so this is the knob that decides how many predictions the task can supply. Sampling
    more than `MAX_SAMPLES` entities buys nothing.

    `max_children_per_parent` bounds the fan-out. RT's sampler already caps how many neighbours
    enter a context (`max_bfs_width`), so a very heavy parent contributes far more rows than the
    model can ever read; capping keeps the database small without changing what a context looks
    like. Set it to 0 to disable.

    `max_rows_per_table` is what actually keeps the sample small. Chasing every parent of every
    kept row re-expands to nearly the whole database on a densely connected schema -- two hops
    through a join table reaches most entities. So each table is capped (seed entities are kept
    first), and any foreign key still pointing outside the sample is set to NULL rather than
    dragging its parent in. `pre` skips null foreign keys, so this costs an edge, never
    correctness -- and it is the alternative to a dangling key, which `pre` would resolve to the
    WRONG parent row without complaining.
    """
    import duckdb

    src_db_dir, src_tasks_dir = Path(src_db_dir), Path(src_tasks_dir)
    out_root = Path(out_root)
    out_db, out_tasks = out_root / "db", out_root / "tasks"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_db.mkdir(parents=True)
    out_tasks.mkdir(parents=True)

    schema = read_schema(src_db_dir)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    t0 = time.time()
    if verbose:
        print(f"[sub] source: {len(schema)} tables, "
              f"{sum(t['n_rows'] for t in schema.values()):,} rows total")

    # ---- 1. seed: the entities each task scores -------------------------------------------- #
    # A task table's own FK metadata says which entity table it is about, so this works for any
    # task without hardcoding entity names.
    keep: dict[str, set] = {t: set() for t in schema}
    seeds: dict[str, set] = {}          # never evicted by the row cap -- these produce predictions
    task_rows: dict[str, dict] = {}
    for tname in task_names:
        tdir = src_tasks_dir / tname
        test_pq = tdir / "test.parquet"
        if not test_pq.exists():
            if verbose:
                print(f"[sub]  {tname}: no test.parquet, skipped")
            continue
        import pyarrow.parquet as pq

        tmeta = _decode_meta(pq.read_schema(test_pq).metadata)
        tfkeys = json.loads(tmeta.get(FKEY_KEY, "{}")) or {}
        if not tfkeys:
            if verbose:
                print(f"[sub]  {tname}: task table declares no foreign key, skipped")
            continue
        picked = {}
        for fk_col, parent in tfkeys.items():
            ids = con.execute(
                f"SELECT DISTINCT \"{fk_col}\" AS id FROM read_parquet('{test_pq.as_posix()}') "
                f"WHERE \"{fk_col}\" IS NOT NULL "
                f"USING SAMPLE {int(n_entities)} ROWS (reservoir, {int(seed)})"
            ).fetchnumpy()["id"]
            if parent in keep:
                keep[parent].update(ids.tolist())
                seeds.setdefault(parent, set()).update(ids.tolist())
            picked[fk_col] = (parent, len(ids))
        task_rows[tname] = {"fkeys": tfkeys, "picked": picked}
        if verbose:
            desc = ", ".join(f"{c}->{p} x{n}" for c, (p, n) in picked.items())
            print(f"[sub]  {tname}: seeded {desc}")

    seeded = {t: len(v) for t, v in keep.items() if v}
    if not seeded:
        raise RuntimeError("no seed entities found -- check task_names and the tasks directory")

    # ---- 2. expand DOWN: rows referencing something already kept ---------------------------- #
    # Monotone growth, bounded by the source tables, so it terminates. Repeated because a child
    # pulled in here can itself be a parent of another table.
    for sweep in range(4):
        grew = False
        for tname, info in schema.items():
            if not info["fkeys"]:
                continue
            clauses = []
            for fk_col, parent in info["fkeys"].items():
                if parent not in keep or not keep[parent]:
                    continue
                ids = ",".join(repr(v) for v in keep[parent])
                clauses.append(f'"{fk_col}" IN ({ids})')
            if not clauses:
                continue
            pkey = info["pkey"]
            if not pkey:
                continue
            cap = ""
            if max_children_per_parent:
                first_fk = next(iter(info["fkeys"]))
                cap = (f" QUALIFY row_number() OVER (PARTITION BY \"{first_fk}\" "
                       f"ORDER BY \"{pkey}\") <= {int(max_children_per_parent)}")
            got = con.execute(
                f'SELECT DISTINCT "{pkey}" AS id FROM read_parquet('
                f"'{Path(info['path']).as_posix()}') WHERE ({' OR '.join(clauses)}){cap}"
            ).fetchnumpy()["id"]
            before = len(keep[tname])
            keep[tname].update(got.tolist())
            grew |= len(keep[tname]) > before
        if not grew:
            break

    # ---- 3. expand UP once, then bound ------------------------------------------------------- #
    # A single sweep, not a fixpoint: chasing parents transitively re-expands to nearly the whole
    # database (two hops through a join table reach most entities), which would defeat the point.
    # Whatever is still unresolved after this gets its foreign key NULLed at write time.
    for tname, info in schema.items():
        if not keep[tname] or not info["fkeys"] or not info["pkey"]:
            continue
        ids = ",".join(repr(v) for v in keep[tname])
        for fk_col, parent in info["fkeys"].items():
            if parent not in keep:
                continue
            got = con.execute(
                f'SELECT DISTINCT "{fk_col}" AS id FROM read_parquet('
                f"'{Path(info['path']).as_posix()}') "
                f'WHERE "{info["pkey"]}" IN ({ids}) AND "{fk_col}" IS NOT NULL'
            ).fetchnumpy()["id"]
            room = max_rows_per_table - len(keep[parent])
            if room > 0:
                keep[parent].update(list(got.tolist())[:room])

    # Cap every table, keeping seed entities first: they are the rows that become predictions, so
    # evicting one would silently shrink the task rather than the context.
    capped = {}
    for tname in keep:
        if len(keep[tname]) > max_rows_per_table:
            s = seeds.get(tname, set()) & keep[tname]
            rest = list(keep[tname] - s)
            keep[tname] = set(list(s)[:max_rows_per_table] + rest[: max_rows_per_table - len(s)])
            capped[tname] = len(keep[tname])
    if capped and verbose:
        print(f"[sub] capped at {max_rows_per_table:,} rows: {capped}")

    # ---- 4. write the sampled tables --------------------------------------------------------- #
    import pyarrow.parquet as _pq

    written, nulled = {}, {}
    for tname, info in schema.items():
        pkey, path = info["pkey"], Path(info["path"]).as_posix()
        # Project every column, rewriting each FK to NULL where its parent was not sampled. This is
        # what makes the bounded sample safe: `pre` skips a null foreign key, but resolves a
        # dangling one to whatever row sits at that offset in the parent.
        cols = _pq.read_schema(info["path"]).names
        proj = []
        for c in cols:
            parent = info["fkeys"].get(c)
            if parent and keep.get(parent):
                pids = ",".join(repr(v) for v in keep[parent])
                proj.append(f'CASE WHEN "{c}" IN ({pids}) THEN "{c}" ELSE NULL END AS "{c}"')
            elif parent:
                proj.append(f'CAST(NULL AS BIGINT) AS "{c}"')
            else:
                proj.append(f'"{c}"')
        select = ", ".join(proj)
        if keep[tname] and pkey:
            ids = ",".join(repr(v) for v in keep[tname])
            sql = f"SELECT {select} FROM read_parquet('{path}') WHERE \"{pkey}\" IN ({ids})"
        elif keep[tname]:
            sql = f"SELECT {select} FROM read_parquet('{path}')"
        else:
            # A table nothing references and that references nothing sampled contributes no
            # context; keeping it empty would break `pre`, so take a small slice instead.
            sql = f"SELECT {select} FROM read_parquet('{path}') LIMIT 1000"
        written[tname] = _write_table(con, sql, info["path"], out_db / f"{tname}.parquet")
        for c, parent in info["fkeys"].items():
            n_null = con.execute(
                f"SELECT count(*) FROM read_parquet("
                f"'{(out_db / f'{tname}.parquet').as_posix()}') WHERE \"{c}\" IS NULL"
            ).fetchone()[0]
            if n_null:
                nulled[f"{tname}.{c}->{parent}"] = int(n_null)

    # ---- 5. write the task tables, restricted to entities that survived ---------------------- #
    task_written = {}
    for tname, tinfo in task_rows.items():
        for split in ("train", "val", "test"):
            src = src_tasks_dir / tname / f"{split}.parquet"
            if not src.exists():
                continue
            clauses = []
            for fk_col, parent in tinfo["fkeys"].items():
                if keep.get(parent):
                    ids = ",".join(repr(v) for v in keep[parent])
                    clauses.append(f'"{fk_col}" IN ({ids})')
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = f"SELECT * FROM read_parquet('{src.as_posix()}'){where}"
            n = _write_table(con, sql, str(src), out_tasks / tname / f"{split}.parquet")
            task_written[f"{tname}/{split}"] = n

    # ---- 6. prove referential integrity ------------------------------------------------------ #
    # The whole point of the closure. A silent dangling key is the one failure mode that produces
    # plausible-looking numbers, so this asserts rather than reports.
    dangling = {}
    for tname, info in schema.items():
        for fk_col, parent in info["fkeys"].items():
            ppk = schema[parent]["pkey"]
            if not ppk:
                continue
            child = (out_db / f"{tname}.parquet").as_posix()
            par = (out_db / f"{parent}.parquet").as_posix()
            n_bad = con.execute(
                f"SELECT count(*) FROM read_parquet('{child}') c "
                f"WHERE c.\"{fk_col}\" IS NOT NULL AND c.\"{fk_col}\" NOT IN "
                f"(SELECT \"{ppk}\" FROM read_parquet('{par}'))"
            ).fetchone()[0]
            if n_bad:
                dangling[f"{tname}.{fk_col}->{parent}"] = int(n_bad)
    if dangling:
        raise RuntimeError(
            f"subsample has dangling foreign keys: {dangling}. `pre` resolves FK values as row "
            f"offsets, so these would silently point at the wrong parent rows."
        )

    report = {
        "tables": written,
        "tasks": task_written,
        "nulled_fkeys": nulled,
        "rows_before": sum(t["n_rows"] for t in schema.values()),
        "rows_after": sum(written.values()),
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"\n[sub] {report['rows_before']:,} -> {report['rows_after']:,} rows "
              f"({100*report['rows_after']/max(report['rows_before'],1):.2f}%) "
              f"in {report['seconds']/60:.1f} min")
        for t, n in sorted(written.items()):
            print(f"    {t:24s} {schema[t]['n_rows']:>12,} -> {n:>9,}")
        print("  task tables:")
        for k, n in sorted(task_written.items()):
            print(f"    {k:32s} {n:>8,}")
        if nulled:
            print("  foreign keys NULLed (parent outside the sample -- edge dropped, never wrong):")
            for k, n in sorted(nulled.items()):
                print(f"    {k:40s} {n:>9,}")
        print("  referential integrity: OK (no dangling foreign keys)")
    return report
