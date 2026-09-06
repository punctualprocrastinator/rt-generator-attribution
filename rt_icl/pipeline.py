"""End-to-end orchestration: generator output -> RelBench export -> RT tasks -> train/val split.

The notebooks are thin wrappers over `convert_all` + `split_tasks` + `rt_icl.prep` + `rt_icl.train`,
so all four generator arms run literally the same code with a different source iterator.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from . import adapters
from .core import (
    canonicalize,
    discover_autocomplete_tasks,
    save_as_relbench,
    validate,
    verify_saved,
)

GENERATORS = ("rdbpfn", "grdm", "reldiff", "plurel")


def db_name_for(generator: str, index: int) -> str:
    """`<gen>-synthetic-<i>`.

    The literal substring "synthetic" matters: `rt.tasks.is_synthetic_db_name` keys off it, and
    `rt.main` uses that to route evaluation to the masked-cell-loss path instead of trying to
    compute AUROC/R2 (which would raise on a single-class or continuous target).
    """
    return f"{generator}-synthetic-{index:05d}"


def source_iterator(generator: str, **kw) -> Iterator[tuple[str, dict, dict]]:
    """Build the (source_id, rdb, kinds) iterator for one generator."""
    if generator == "rdbpfn":
        return adapters.iter_rdbpfn(kw["root"], limit=kw.get("limit"))
    if generator == "grdm":
        return adapters.iter_grdm(
            kw["root"],
            schema_dir=kw.get("schema_dir"),
            limit=kw.get("limit"),
            allow_inferred_schema=kw.get("allow_inferred_schema", False),
        )
    if generator == "reldiff":
        return adapters.iter_reldiff(
            kw["root"],
            metadata_path=kw.get("metadata_path"),
            limit=kw.get("limit"),
            interleave=kw.get("interleave", True),
        )
    if generator == "plurel":
        return adapters.iter_plurel(
            num_dbs=kw.get("num_dbs", 1000),
            seed_offset=kw.get("seed_offset", 0),
            cache_dir=kw.get("cache_dir"),
            limit=kw.get("limit"),
        )
    raise ValueError(f"unknown generator {generator!r}; expected one of {GENERATORS}")


def convert_all(
    generator: str,
    source: Iterator[tuple[str, dict, dict]],
    *,
    target_dbs: int = 1000,
    tasks_per_db: int = 4,
    home: str | None = None,
    min_tables: int = 2,
    min_non_missing: int = 64,
    max_fkeys: int = 5,
    max_text_nunique: int = 512,
    progress_every: int = 25,
    manifest_path: str | Path | None = None,
) -> dict:
    """Convert up to `target_dbs` source databases into RelBench exports + RT task tuples.

    A database is kept only if it survives canonicalization with >= `min_tables` tables, has no
    table with more than `max_fkeys` foreign keys (RT's fixed-width F->P slot), AND yields at least
    one maskable (boolean or numeric) target column -- a database with no such column cannot
    contribute a training signal.
    """
    kept, skipped = [], []
    agg_report: dict = {}
    t0 = time.time()

    for src_id, rdb, kinds in source:
        if len(kept) >= target_dbs:
            break
        rep: dict = {}
        try:
            rdb = canonicalize(
                rdb, kinds=kinds, max_text_nunique=max_text_nunique, report=rep
            )
            if len(rdb) < min_tables:
                skipped.append({"source": src_id, "why": f"{len(rdb)} tables < {min_tables}"})
                continue

            # RT can only carry 5 foreign->primary neighbours per row (MAX_F2P_NBRS in the Rust
            # sampler). Databases exceeding that cannot be represented at all -- catch it here,
            # where it is one counted skip, instead of at step 0 of training where it appears as
            # an opaque "DataLoader worker exited unexpectedly".
            over = {n: len(t.fkey_col_to_pkey_table) for n, t in rdb.items()
                    if len(t.fkey_col_to_pkey_table) > max_fkeys}
            if over:
                worst = max(over.values())
                skipped.append({
                    "source": src_id,
                    "why": f"table with >{max_fkeys} foreign keys",
                    "detail": (
                        f"{len(over)} of {len(rdb)} tables exceed MAX_F2P_NBRS={max_fkeys} "
                        f"(max {worst} FKs): {dict(list(over.items())[:5])}"
                    ),
                })
                continue
            validate(rdb, max_fkeys=max_fkeys)
            index = len(kept)
            db_name = db_name_for(generator, index)
            tasks = discover_autocomplete_tasks(
                rdb,
                db_name,
                tasks_per_db=tasks_per_db,
                min_non_missing=min_non_missing,
                seed=index,
            )
            if not tasks:
                # Say WHY there was nothing to mask -- a bare "no maskable target column" cannot
                # distinguish "this database really is all-categorical" from "the files were
                # truncated and every column came back too short to qualify".
                import pandas as _pd

                comp: dict[str, int] = {}
                rows = []
                for _t in rdb.values():
                    rows.append(len(_t.df))
                    for _c in _t.feature_cols():
                        _s = _t.df[_c]
                        k = (
                            "boolean" if str(_s.dtype) == "boolean"
                            else "numeric" if _pd.api.types.is_numeric_dtype(_s)
                            else "datetime" if _pd.api.types.is_datetime64_any_dtype(_s)
                            else "text"
                        )
                        comp[k] = comp.get(k, 0) + 1
                skipped.append(
                    {
                        "source": src_id,
                        "why": "no maskable target column",
                        "detail": (
                            f"tables={len(rdb)} rows(min/max)={min(rows)}/{max(rows)} "
                            f"feature_kinds={comp or 'none'} "
                            f"(need a boolean or numeric column with >= {min_non_missing} "
                            f"non-missing values)"
                        ),
                    }
                )
                continue
            save_as_relbench(rdb, db_name, home=home, source=src_id)
            info = verify_saved(db_name, home=home)
        except Exception as e:  # noqa: BLE001 - one bad database must not kill a 1000-db run
            skipped.append({"source": src_id, "why": f"{type(e).__name__}: {e}"})
            continue

        for k, v in rep.items():
            if isinstance(v, list):
                agg_report[k] = agg_report.get(k, 0) + len(v)
            elif isinstance(v, dict):
                sub = agg_report.setdefault(k, {})
                for kk, vv in v.items():
                    sub[kk] = sub.get(kk, 0) + vv
            else:
                agg_report[k] = agg_report.get(k, 0) + v

        kept.append(
            {
                "db_name": db_name,
                "source": src_id,
                "n_tables": len(rdb),
                "n_rows": info["rows"],
                "n_cells": info["cells"],
                "tasks": [
                    {
                        "table": t[1],
                        "target": t[2],
                        "kind": "clf"
                        if str(rdb[t[1]].df[t[2]].dtype) == "boolean"
                        else "reg",
                    }
                    for t in tasks
                ],
            }
        )
        if len(kept) % progress_every == 0:
            el = time.time() - t0
            print(
                f"[convert] {len(kept)}/{target_dbs} kept, {len(skipped)} skipped, "
                f"{el/60:.1f} min ({len(kept)/max(el,1e-9):.1f} db/s)",
                flush=True,
            )

    n_clf = sum(1 for d in kept for t in d["tasks"] if t["kind"] == "clf")
    n_reg = sum(1 for d in kept for t in d["tasks"] if t["kind"] == "reg")
    manifest = {
        "generator": generator,
        "n_databases": len(kept),
        "n_tasks": n_clf + n_reg,
        "n_clf_tasks": n_clf,
        "n_reg_tasks": n_reg,
        "total_rows": sum(d["n_rows"] for d in kept),
        "total_cells": sum(d["n_cells"] for d in kept),
        "canonicalize_report": agg_report,
        "databases": kept,
        "skipped": skipped,
        "convert_seconds": time.time() - t0,
    }
    print(
        f"\n[convert] DONE {len(kept)} databases, {n_clf + n_reg} tasks "
        f"(clf {n_clf} / reg {n_reg}), {manifest['total_rows']:,} rows, "
        f"{manifest['total_cells']:,} cells, {len(skipped)} skipped"
    )
    if skipped:
        from collections import Counter

        why = Counter(s["why"].split(":")[0] for s in skipped)
        print(f"[convert] skip reasons: {dict(why)}")
        for s_ in skipped[:3]:
            if s_.get("detail"):
                print(f"    e.g. {s_['source']}: {s_['detail']}")
    if manifest_path:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(manifest, indent=2, default=str))
        print(f"[convert] manifest -> {manifest_path}")
    return manifest


def split_tasks(
    manifest: dict, *, n_val_dbs: int = 0, eval_tasks_cap: int = 12, seed: int = 0
) -> tuple[list[tuple], list[tuple]]:
    """Build the pretraining task list (and, optionally, a held-out-database probe).

    **The default is no held-out split.** These synthetic databases exist to PRETRAIN the model on
    as much relational structure as possible; measuring the model is a separate downstream step
    (few-shot / in-context inference on real benchmarks) that consumes the saved weights. Holding
    databases back here would only shrink the pretraining corpus for a number we do not need.

    Set `n_val_dbs > 0` if you want a validation curve during pretraining. It is then a *training
    diagnostic* on held-out synthetic databases, not an evaluation of the study. Keep
    `eval_tasks_cap` small: `rt.main` builds one DataLoader per eval task with
    `persistent_workers=True`, so each one costs `num_workers` resident processes.
    """
    import random

    dbs = list(manifest["databases"])
    rng = random.Random(seed)

    def tuples(entries):
        return [(d["db_name"], t["table"], t["target"], []) for d in entries for t in d["tasks"]]

    if not n_val_dbs:
        train_tasks = tuples(dbs)
        print(
            f"[split] pretraining on ALL {len(dbs)} databases -> {len(train_tasks)} masked-cell "
            f"tasks (no held-out split; evaluation happens downstream on the saved weights)"
        )
        return train_tasks, []

    rng.shuffle(dbs)
    n_val = min(n_val_dbs, max(1, len(dbs) // 10))
    val_dbs, train_dbs = dbs[:n_val], dbs[n_val:]
    train_tasks = tuples(train_dbs)
    eval_all = tuples(val_dbs)
    rng.shuffle(eval_all)
    eval_tasks = eval_all[:eval_tasks_cap]
    print(
        f"[split] {len(train_dbs)} train databases -> {len(train_tasks)} tasks | "
        f"{len(val_dbs)} held-out databases -> {len(eval_tasks)} probe tasks "
        f"(capped from {len(eval_all)}) -- a training diagnostic, not the study's evaluation"
    )
    return train_tasks, eval_tasks


def all_db_names(manifest: dict) -> list[str]:
    return [d["db_name"] for d in manifest["databases"]]
