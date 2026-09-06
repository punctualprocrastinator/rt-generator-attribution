"""Benchmark pretrained RT checkpoints on held-out RelBench tasks, across context lengths.

This answers study RQ1 ("which generator gives the most accurate learning to an RFM model?") by
running each arm's checkpoint over the same RelBench tasks with no fine-tuning -- the zero-shot
prompting setting the RT paper reports -- and sweeping the context length to see how much each
pretrained model benefits from more in-context evidence.

Contamination
-------------
GRDM and RelDiff generate *from* RelBench references: their corpora are built from the 35% temporal
slice of rel-f1, rel-hm, rel-avito, rel-event and rel-trial. Evaluating those arms on the same
databases would score them on data derived from their own training material. RDB-PFN and PluRel
generate from scratch and are clean everywhere.

RelBench **v2** ships 11 `rel-*` databases, so removing those five still leaves six untouched ones:
rel-amazon, rel-arxiv, rel-mimic, rel-ratebeer, rel-salt and rel-stack. Those are the default.
`CONTAMINATED` records the rest so a deliberate comparison on them stays labelled.

Context length
--------------
RT has no positional encodings, so it accepts any context length; the pretraining value (1024) is
not a hard limit. But `RelationalTransformer.forward` materializes several dense (B, S, S) masks and
two (B, S, S, 5) intermediates to build them, so memory grows with B*S^2. At S=30000 those
intermediates are ~4.5 GB EACH at batch size 1. `plan_batch_size` scales the batch down accordingly
and `evaluate_task` records an OOM instead of crashing the sweep.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

# Databases each generator's corpus was derived from; empty means generated from scratch.
CONTAMINATED: dict[str, tuple[str, ...]] = {
    "grdm": ("rel-f1", "rel-hm", "rel-avito", "rel-event", "rel-trial"),
    "reldiff": ("rel-f1", "rel-hm", "rel-avito", "rel-event", "rel-trial"),
    "rdbpfn": (),
    "plurel": (),
}

#: RelBench **v2** has 11 `rel-*` databases. Removing the five the generators were built from
#: leaves exactly these six, which no arm has seen.
HELD_OUT_DBS: tuple[str, ...] = (
    "rel-amazon", "rel-arxiv", "rel-mimic", "rel-ratebeer", "rel-salt", "rel-stack",
)

#: Tasks RT can actually score. Its decoders are boolean and numeric only, so
#: BINARY_CLASSIFICATION -> AUROC and REGRESSION -> R2. RelBench v2's MULTICLASS_CLASSIFICATION
#: and LINK_PREDICTION tasks have no corresponding head and are excluded -- see UNSCORABLE.
#: (database, task, target column, kind, leakage columns to drop)
ENTITY_TASKS: list[tuple[str, str, str, str, list[str]]] = [
    # -- rel-amazon --
    ("rel-amazon", "user-churn", "churn", "clf", []),
    ("rel-amazon", "item-churn", "churn", "clf", []),
    ("rel-amazon", "user-ltv", "ltv", "reg", []),
    ("rel-amazon", "item-ltv", "ltv", "reg", []),
    ("rel-amazon", "review-rating", "rating", "reg", ["review_text", "summary"]),
    # -- rel-arxiv --
    ("rel-arxiv", "paper-citation", "cited", "clf", []),
    ("rel-arxiv", "author-publication", "publication_count", "reg", []),
    # -- rel-mimic --
    ("rel-mimic", "patient-iculengthofstay", "los_icu_binary", "clf", []),
    # -- rel-ratebeer --
    ("rel-ratebeer", "beer-churn", "rating_churn", "clf", []),
    ("rel-ratebeer", "user-churn", "user_churn", "clf", []),
    ("rel-ratebeer", "brewer-dormant", "dormant", "clf", []),
    ("rel-ratebeer", "user-count", "num_ratings", "reg", []),
    # -- rel-stack --
    ("rel-stack", "user-engagement", "contribution", "clf", []),
    ("rel-stack", "user-badge", "WillGetBadge", "clf", []),
    ("rel-stack", "post-votes", "popularity", "reg", []),
]

#: Recorded so the gap is explicit rather than silent. rel-salt contributes NOTHING scorable:
#: all eight of its tasks are multiclass.
UNSCORABLE: dict[str, str] = {
    "rel-salt/*": "all 8 tasks are MULTICLASS_CLASSIFICATION -- RT has no class head",
    "rel-ratebeer/beer_ratings-total_score": (
        "unavoidable leakage: total_score is composed of aroma/flavor/palate/appearance/overall, "
        "which live on the SOURCE ratings table and are reachable through relational context. "
        "rt.data resolves drop columns against the TASK table only (data.py:82), so they cannot "
        "be dropped; removing them from the source table instead would strip information from the "
        "four other rel-ratebeer tasks. Excluded rather than reported as a leaky R2."
    ),
    "rel-arxiv/author-category": "MULTICLASS_CLASSIFICATION",
    "rel-stack/badges-class": "MULTICLASS_CLASSIFICATION",
    "rel-amazon/user-item-*": "LINK_PREDICTION -- RT has no ranking head",
    "rel-stack/user-post-comment": "LINK_PREDICTION",
    "rel-stack/post-post-related": "LINK_PREDICTION",
    "rel-ratebeer/user-*-liked, user-beer-favorite": "LINK_PREDICTION",
    "rel-arxiv/paper-paper-cocitation": "LINK_PREDICTION",
}

#: The sweep requested for the study.
CTX_LENS: tuple[int, ...] = (100, 200, 512, 1024, 30000)


#: (database, task) -> target column, for every task we score as classification.
CLF_TARGETS: dict[tuple[str, str], str] = {
    (db, task): col for db, task, col, kind, _ in ENTITY_TASKS if kind == "clf"
}


def tasks_for(dbs: list[str] | tuple[str, ...] | None = None) -> list[tuple]:
    """Entity tasks for the given databases (default: the held-out ones).

    A subsampled database (`rel-amazon-sub`) inherits its source's task list under the new name, so
    it can be requested exactly like any other database.
    """
    want = tuple(dbs) if dbs else HELD_OUT_DBS
    known = {t[0] for t in ENTITY_TASKS}
    out: list[tuple] = []
    for d in want:
        if d in known:
            out.extend(t for t in ENTITY_TASKS if t[0] == d)
            continue
        # longest match, so "rel-amazon-sub" resolves to rel-amazon and never to a shorter prefix
        src = max((k for k in known if d.startswith(k + "-")), key=len, default=None)
        if src is None:
            raise ValueError(f"unknown database {d!r}; known: {sorted(known)}")
        out.extend(subsampled_tasks(src, d))
    if not out:
        raise ValueError(f"no entity tasks for {want}; known databases: {sorted(known)}")
    return out


def contamination_report(generators: list[str], dbs: list[str] | tuple[str, ...]) -> dict:
    """Which (generator, database) pairs are contaminated. Empty `flagged` means a clean sweep."""
    flagged = {
        g: [d for d in dbs if d in CONTAMINATED.get(g, ())]
        for g in generators
    }
    flagged = {g: v for g, v in flagged.items() if v}
    return {"flagged": flagged, "clean": not flagged}


# --------------------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------------------- #
def prepare_relbench_db(
    db_name: str,
    task_names: list[str],
    binary: str | Path,
    *,
    home: str | None = None,
    embedding_model: str = "all-MiniLM-L12-v2",
    force: bool = False,
) -> dict:
    """Materialize a RelBench database + its task tables, then preprocess for RT.

    Writes `<relbench cache>/<db>/db/*.parquet` and `.../tasks/<task>/{train,val,test}.parquet`,
    which is exactly the layout `rustler pre` expects, then runs the preprocessor and the text
    embedder. Downloads are large (rel-amazon is several GB), so this is cached and resumable.
    """
    import gc

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from . import prep

    def _rss() -> str:
        ram = prep.system_ram_gb()
        return f" [RAM {ram:.0f} GB total]" if ram else ""

    h = Path(home or os.environ["HOME"])
    root = h / "scratch" / "relbench" / db_name
    if force or not (root / "db").is_dir():
        print(f"[bench] downloading {db_name} ...{_rss()}", flush=True)
        ds = get_dataset(db_name, download=True)
        db = ds.get_db(upto_test_timestamp=False)  # test split needs the full history
        db.save(root / "db")
        n_tables = len(db.table_dict)
        # Drop the in-memory copy NOW. It is on disk, and everything after this point -- the task
        # tables, and especially the `pre` subprocess, which reads the whole database back from
        # parquet -- runs alongside it. Holding it here doubles peak RAM for no benefit, and on the
        # large databases that is the difference between finishing and the kernel being killed.
        del db, ds
        gc.collect()
        print(f"[bench] {db_name}: {n_tables} tables -> {root/'db'} (in-memory copy released)")

    for tname in task_names:
        out = root / "tasks" / tname
        if not force and (out / "test.parquet").exists():
            continue
        task = get_task(db_name, tname, download=True)
        target = CLF_TARGETS.get((db_name, tname))
        for split in ("train", "val", "test"):
            tbl = task.get_table(split, mask_input_cols=False)
            # Force classification labels to BOOLEAN before `pre` ever sees them.
            #
            # `pre.rs` casts these per-task by hand, keyed on RelBench **v1** task-table names, so a
            # v2 task whose name differs (rel-ratebeer/beer-churn) or a v2-only database with no
            # block at all (rel-arxiv) keeps an integer label. RT then treats that cell as NUMERIC:
            # the boolean head gets no label, every y reads as 0, and AUROC comes out nan with
            # status="single-class". Casting here is name-independent -- it uses OUR task list, the
            # same one that decides the cell is scored as classification -- so it cannot drift out
            # of sync with rustler's hardcoded names again.
            if target and target in tbl.df.columns:
                s = tbl.df[target]
                if str(s.dtype) not in ("bool", "boolean"):
                    # nullable "boolean" only when nulls exist: astype(bool) would turn NaN into
                    # True, inventing positive labels
                    tbl.df[target] = s.astype("boolean") if s.isna().any() else s.astype(bool)
                    print(f"[bench] {db_name}/{tname}: cast {target} {s.dtype} -> "
                          f"{tbl.df[target].dtype}")
            tbl.save(out / f"{split}.parquet")
            del tbl
        del task
        gc.collect()
        print(f"[bench] {db_name}/{tname}: task tables -> {out}")
    gc.collect()

    if force or not prep.pre_done(db_name, home=home):
        fails = prep.run_pre([db_name], binary, home=home, jobs=1, progress_every=1)
        if fails:
            # Print in full rather than truncating: the exit status is at the TOP of this report
            # and is the part that distinguishes "killed" from "crashed", so a tail-slice would
            # cut off exactly the useful half.
            print(f"\n=== `pre` failed for {db_name} ===\n{list(fails.values())[0]}", flush=True)
            raise RuntimeError(f"`pre` failed for {db_name} -- see the report above")
    # embed_all reports WHY it failed; verify_embeddings only sees that a file is absent. Raising
    # on the former keeps the cause ("CUDA out of memory", "no text.json") instead of replacing it
    # with "missing files", which describes the symptom and names no fix.
    efail = prep.embed_all(
        [db_name], embedding_model=embedding_model, home=home, progress_every=1
    )
    if efail:
        raise RuntimeError(f"embedding failed for {db_name}: {efail[db_name]}")
    bad = prep.verify_embeddings([db_name], embedding_model, home=home)
    if bad:
        raise RuntimeError(
            f"embeddings bad for {db_name}: {bad}. embed_all reported no error, so the file is "
            f"present but the wrong size -- delete ~/scratch/pre/{db_name}/text_emb_*.bin and rerun."
        )
    return {"db": db_name, "tasks": task_names, "root": str(root)}


def _source_tar(archive_dir: str | Path, db_name: str) -> Path:
    return Path(archive_dir) / f"{db_name}_src.tar"


def archive_source_export(db_name: str, archive_dir: str | Path, *, home: str | None = None):
    """Archive `scratch/relbench/<db>` -- the exported parquet `pre` reads.

    The pre/<db>.tar on Drive holds only the PREPROCESSED output, so re-running `pre` in a later
    session (any change to how task tables are written, e.g. a label dtype) means downloading the
    database again from scratch. Keeping the source export makes that a restore instead.
    """
    import tarfile

    h = Path(home or os.environ["HOME"])
    src = h / "scratch" / "relbench" / db_name
    if not (src / "db").is_dir():
        return None
    dest = _source_tar(archive_dir, db_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with tarfile.open(tmp, "w") as tf:
        tf.add(src, arcname=db_name)
    tmp.replace(dest)  # atomic: a truncated tar must never look complete
    print(f"[bench] {db_name}: source export archived -> {dest} "
          f"({dest.stat().st_size/1e9:.2f} GB) -- future rebuilds skip the download")
    return dest


def restore_source_export(db_name: str, archive_dir: str | Path, *, home: str | None = None) -> bool:
    """Restore the source export from Drive. True if `pre`'s input is now on local disk."""
    import tarfile

    h = Path(home or os.environ["HOME"])
    tar = _source_tar(archive_dir, db_name)
    if not tar.exists():
        return False
    root = h / "scratch" / "relbench"
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        tf.extractall(root)
    ok = (root / db_name / "db").is_dir()
    print(f"[bench] {db_name}: source export restored from Drive"
          f"{'' if ok else ' (INCOMPLETE)'} -- no download needed")
    return ok


def relbench_cache_db_dir(db_name: str) -> Path:
    """Where relbench unzips a database's parquet."""
    return Path(os.environ.get("HOME", "/root")) / ".cache" / "relbench" / db_name / "db"


def ensure_downloaded(db_name: str, verbose: bool = True) -> Path:
    """Get `<cache>/<db>/db/*.parquet` on disk WITHOUT loading the database into this process.

    `Dataset.get_db()` materializes every table as a pandas frame, which is precisely what rel-amazon
    cannot do here. But the download-and-unzip happens *before* that load, so running the loader in a
    CHILD process gets us the side effect we need and confines the memory blow-up: if the child is
    OOM-killed the parquet is already on disk and the notebook kernel survives.
    """
    import subprocess
    import sys

    d = relbench_cache_db_dir(db_name)
    if d.is_dir() and any(d.glob("*.parquet")):
        if verbose:
            print(f"[bench] {db_name}: source parquet already cached at {d}")
        return d
    if verbose:
        print(f"[bench] {db_name}: downloading (in a child process, so its in-memory load "
              f"cannot take down this kernel) ...", flush=True)
    subprocess.run(
        [sys.executable, "-c",
         f"from relbench.datasets import get_dataset; "
         f"get_dataset({db_name!r}, download=True).get_db(upto_test_timestamp=False)"],
        check=False,   # an OOM in the child is EXPECTED and harmless: we want only the unzip
    )
    if not (d.is_dir() and any(d.glob("*.parquet"))):
        raise RuntimeError(
            f"{db_name}: download did not produce parquet at {d}. Check disk space -- the archive "
            f"and the unzipped copy both have to fit."
        )
    if verbose:
        print(f"[bench] {db_name}: source parquet ready at {d}")
    return d


def prepare_subsampled_db(
    src_db: str,
    out_name: str,
    task_names: list[str],
    binary: str | Path,
    *,
    home: str | None = None,
    n_entities: int = 3000,
    max_children_per_parent: int = 200,
    max_rows_per_table: int = 300_000,
    embedding_model: str = "all-MiniLM-L12-v2",
    seed: int = 0,
) -> dict:
    """Build `out_name` as a bounded subsample of `src_db`, then preprocess it like any database.

    For databases that do not fit end to end. The result is a real RelBench export under a DISTINCT
    name, so a subsampled score can never be mistaken for a full-database one -- and because all
    four generator arms are scored on the same prepared copy, within-task comparisons stay valid.
    """
    from relbench.tasks import get_task

    from . import prep, subsample

    h = Path(home or os.environ["HOME"])
    src_db_dir = ensure_downloaded(src_db)

    # Task tables are orders of magnitude smaller than the database, so these can be staged in
    # process -- one split at a time, with the classification labels cast as everywhere else.
    stage = h / "scratch" / "relbench" / f"_stage_{src_db}" / "tasks"
    for tname in task_names:
        out = stage / tname
        if (out / "test.parquet").exists():
            continue
        task = get_task(src_db, tname, download=True)
        target = CLF_TARGETS.get((src_db, tname))
        for split in ("train", "val", "test"):
            tbl = task.get_table(split, mask_input_cols=False)
            if target and target in tbl.df.columns:
                s = tbl.df[target]
                if str(s.dtype) not in ("bool", "boolean"):
                    tbl.df[target] = s.astype("boolean") if s.isna().any() else s.astype(bool)
            tbl.save(out / f"{split}.parquet")
            del tbl
        del task
        print(f"[bench] {src_db}/{tname}: task tables staged")

    root = h / "scratch" / "relbench" / out_name
    report = subsample.subsample_db(
        src_db_dir, stage, root, task_names,
        n_entities=n_entities, max_children_per_parent=max_children_per_parent,
        max_rows_per_table=max_rows_per_table, seed=seed,
    )
    (root / "rt_icl_subsample.json").write_text(json.dumps(
        {"source": src_db, "n_entities": n_entities,
         "max_children_per_parent": max_children_per_parent,
         "max_rows_per_table": max_rows_per_table, "seed": seed, **report},
        indent=2, default=str))

    fails = prep.run_pre([out_name], binary, home=home, jobs=1, progress_every=1)
    if fails:
        print(f"\n=== `pre` failed for {out_name} ===\n{list(fails.values())[0]}", flush=True)
        raise RuntimeError(f"`pre` failed for {out_name} -- see the report above")
    efail = prep.embed_all([out_name], embedding_model=embedding_model, home=home, progress_every=1)
    if efail:
        raise RuntimeError(f"embedding failed for {out_name}: {efail[out_name]}")
    bad = prep.verify_embeddings([out_name], embedding_model, home=home)
    if bad:
        raise RuntimeError(f"embeddings bad for {out_name}: {bad}")
    return {"db": out_name, "source": src_db, "report": report}


def subsampled_tasks(src_db: str, out_name: str) -> list[tuple]:
    """`src_db`'s entity tasks, re-pointed at the subsampled database."""
    return [(out_name, *rest) for db, *rest in ENTITY_TASKS if db == src_db]


def prepare_or_restore(
    db_name: str,
    task_names: list[str],
    binary: str | Path,
    archive_dir: str | Path,
    *,
    home: str | None = None,
    embedding_model: str = "all-MiniLM-L12-v2",
    save: bool = True,
    force: bool = False,
    archive_source: bool = True,
) -> dict:
    """Get `db_name` ready to evaluate, preferring a prepared tar on Drive over redoing the work.

    Preparation (download, `pre`, MiniLM embeddings) is the expensive half of benchmarking and it
    lands in the ephemeral `~/scratch`, so a session that dies repeats hours of it. Archiving each
    database separately lets preparation run once on a cheap GPU and every later evaluation session
    -- however it is sharded -- restore only what it needs in a couple of minutes.
    """
    from . import prep

    tar = Path(archive_dir) / f"{db_name}.tar"
    if force:
        # Rebuild from the task tables up: re-export the task tables, re-run `pre`, re-embed.
        #
        # The exported database under scratch/relbench/<db>/db is KEPT, so in the session that
        # prepared it nothing is downloaded again. Across sessions that is no help: ~/scratch and
        # ~/.cache/relbench are both on the ephemeral VM disk, and only the pre/<db>.tar on Drive
        # survives -- and that holds the PREPROCESSED output, not the source parquet `pre` needs.
        # So a forced rebuild in a fresh session re-downloads. Prefer to force in the session that
        # still has the export, or archive the source too (see `archive_source`).
        h = Path(home or os.environ["HOME"])
        shutil.rmtree(h / "scratch" / "pre" / db_name, ignore_errors=True)
        shutil.rmtree(h / "scratch" / "relbench" / db_name / "tasks", ignore_errors=True)
        tar.unlink(missing_ok=True)
        print(f"[bench] {db_name}: forced rebuild")
        h = Path(home or os.environ["HOME"])
        if not (h / "scratch" / "relbench" / db_name / "db").is_dir():
            # fresh session: the export died with the last VM. Restore it if we kept one.
            if not restore_source_export(db_name, archive_dir, home=home):
                print(f"[bench] {db_name}: no source export on Drive -- the database has to be "
                      f"downloaded again. Set ARCHIVE_SOURCE=True so the next rebuild does not.")
    elif prep.pre_done(db_name, home=home) and prep.verify_embeddings(
        [db_name], embedding_model, home=home
    ) == {}:
        return {"db": db_name, "source": "local", "archive": str(tar)}
    if not force and tar.exists():
        prep.restore_pre(tar, home=home)
        bad = prep.verify_embeddings([db_name], embedding_model, home=home)
        if not bad:
            return {"db": db_name, "source": "archive", "archive": str(tar)}
        print(f"[bench] {db_name}: archive restored but embeddings incomplete ({bad}); rebuilding")

    prepare_relbench_db(db_name, task_names, binary, home=home,
                        embedding_model=embedding_model)
    if save:
        prep.archive_pre([db_name], tar, home=home)
        if archive_source:
            archive_source_export(db_name, archive_dir, home=home)
    return {"db": db_name, "source": "built", "archive": str(tar)}


# --------------------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------------------- #
#: Fraction of free VRAM a single cell is allowed to plan for. The rest absorbs the model, the
#: CUDA context, allocator fragmentation and the decoder activations.
MEM_HEADROOM = 0.55


def bytes_per_seq(ctx_len: int, lean_masks: bool = False) -> int:
    """Roughly what one sequence in the batch costs, in bytes, at this context length.

    Stock: the mask machinery dominates and is quadratic -- three kept (B, S, S) bool masks, a few
    live temporaries, and one (B, S, S, 5) intermediate.

    Lean: no dense mask exists. What remains is the BlockMask index arrays (quadratic but in
    128-wide *blocks*, so ~16000x smaller) and the activations, which are linear in S.
    """
    if not lean_masks:
        return ctx_len ** 2 * (6 + 5)
    blocks = math.ceil(ctx_len / 128) ** 2 * 4 * 3      # 3 masks of int32 block indices
    acts = ctx_len * 256 * 2 * 24                       # d_model 256, bf16, ~24 live tensors
    return blocks + acts


def free_vram_gb() -> float | None:
    """Free VRAM in GB, or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()[0] / 1e9
    except Exception:  # noqa: BLE001 - a memory probe must never break the sweep
        return None


def plan_batch_size(
    ctx_len: int,
    base_batch: int = 64,
    base_ctx: int = 1024,
    *,
    lean_masks: bool = False,
    free_gb: float | None = None,
) -> int:
    """Batch size for one context length, under whichever mask path is active.

    Stock: memory is dominated by the dense masks and grows with B*S^2, so the batch has to fall
    quadratically -- which is what forces batch 1 at 30000 and an A100 to hold it.

    Lean (`compat.patch_lean_block_masks`): the dense masks are gone, so the binding cost is the
    activations, linear in B*S. Holding B*S constant instead of B*S^2 lets long contexts keep a
    usable batch. Shorter lengths are unaffected -- the result is still capped at `base_batch`.

    `free_gb` additionally caps the batch to what actually fits on this card, which is what lets a
    24 GB L4 stand in for a 40 GB A100 instead of guessing and OOMing.
    """
    if lean_masks:
        b = int(base_batch * base_ctx / max(ctx_len, 1))
    else:
        b = int(base_batch * (base_ctx / max(ctx_len, 1)) ** 2)
    b = max(1, min(base_batch, b))
    if free_gb is not None:
        per = bytes_per_seq(ctx_len, lean_masks) / 1e9
        b = max(1, min(b, int(free_gb * MEM_HEADROOM / max(per, 1e-9))))
    return b


def unwrap_state_dict(sd: dict) -> tuple[dict, list[str]]:
    """Strip `torch.compile` / DDP key prefixes from a checkpoint.

    Pretraining runs `rt.main` with `compile_=True`, and `torch.compile` wraps the module, so every
    saved key is prefixed `_orig_mod.` (DDP adds `module.`). Loading those into a plain
    `RelationalTransformer` reports every parameter as both missing AND unexpected.

    Stripping is a pure rename -- tensors, shapes and order are untouched -- and only happens when
    EVERY key shares the prefix, so a genuine architecture mismatch still fails loudly instead of
    being papered over.
    """
    stripped: list[str] = []
    changed = True
    while changed and sd:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if all(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix):]: v for k, v in sd.items()}
                stripped.append(prefix)
                changed = True
    return sd, stripped


def load_model(ckpt_path: str | Path, device: str = "cuda", arch: dict | None = None):
    """Load a pretrained RT checkpoint in the architecture the study pins."""
    import torch

    from rt.model import RelationalTransformer

    from .train import PAPER_ARCH

    a = dict(PAPER_ARCH)
    if arch:
        a.update(arch)
    net = RelationalTransformer(
        num_blocks=a["num_blocks"], d_model=a["d_model"], d_text=a["d_text"],
        num_heads=a["num_heads"], d_ff=a["d_ff"],
    )
    sd = torch.load(Path(ckpt_path).expanduser(), map_location="cpu")
    # some savers nest the weights under a key rather than storing them at the top level
    if isinstance(sd, dict) and not any(torch.is_tensor(v) for v in sd.values()):
        for key in ("state_dict", "model", "net"):
            if isinstance(sd.get(key), dict):
                sd = sd[key]
                break
    sd, stripped = unwrap_state_dict(sd)
    if stripped:
        print(f"[bench] {Path(ckpt_path).name}: stripped {''.join(stripped)} key prefix "
              f"(checkpoint was saved from a compiled model)")
    net.load_state_dict(sd)
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    return net.to(device).to(dt).eval()


def evaluate_task(
    net,
    db_name: str,
    task_name: str,
    target_col: str,
    kind: str,
    *,
    ctx_len: int,
    batch_size: int | None = None,
    lean_masks: bool = False,
    oom_backoff: bool = True,
    **kw,
) -> dict:
    """Score one cell, halving the batch and retrying if it does not fit.

    Without this an OOM loses the whole cell, and on a smaller card that is most of the long-context
    sweep. Batch size does not affect the metric -- only how many sequences are scored at once -- so
    retrying smaller is a free recovery rather than a change of measurement. `batch_size` and
    `oom_retries` are recorded, so a cell that had to back off is visible in the results.
    """
    bs = batch_size or plan_batch_size(
        ctx_len, lean_masks=lean_masks, free_gb=free_vram_gb()
    )
    retries = 0
    while True:
        rec = _evaluate_once(
            net, db_name, task_name, target_col, kind,
            ctx_len=ctx_len, batch_size=bs, **kw,
        )
        if rec.get("status") != "OOM" or not oom_backoff or bs <= 1:
            rec["oom_retries"] = retries
            return rec
        bs = max(1, bs // 2)
        retries += 1
        print(f"[bench]   OOM at batch {bs * 2} -> retrying at {bs}", flush=True)


def _evaluate_once(
    net,
    db_name: str,
    task_name: str,
    target_col: str,
    kind: str,
    *,
    ctx_len: int,
    drops: list[str] | None = None,
    split: str = "test",
    batch_size: int | None = None,
    max_samples: int = 2048,
    max_batches: int = 100_000,
    max_bfs_width: int = 128,
    num_workers: int = 2,
    embedding_model: str = "all-MiniLM-L12-v2",
    d_text: int = 384,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Score one checkpoint on one task at one context length. Never raises: OOM is a result.

    The cap is on PREDICTIONS (`max_samples`), not batches. Batch size shrinks as context grows, so
    a fixed batch cap would silently give the long-context runs far fewer predictions and wider
    error bars than the short ones. Every row records `n`, so any remaining difference is visible.
    """
    import torch
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

    from rt.data import RelationalDataset

    bs = batch_size or plan_batch_size(ctx_len)
    rec = {
        "db": db_name, "task": task_name, "target": target_col, "kind": kind,
        "ctx_len": ctx_len, "batch_size": bs, "split": split,
    }
    loader = None
    t0 = time.time()
    try:
        ds = RelationalDataset(
            tasks=[(db_name, task_name, target_col, split, list(drops or []))],
            batch_size=bs, rank=0, world_size=1,
            ctx_len=ctx_len, max_bfs_width=max_bfs_width,
            embedding_model=embedding_model, d_text=d_text, seed=seed,
        )
        ds.sampler.shuffle_py(0)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers,
            pin_memory=(device == "cuda"), in_order=True,
        )
        preds, labels = [], []
        n_seen = 0
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= max_batches or n_seen >= max_samples:
                    break
                tbs = batch.pop("true_batch_size")
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                # mask the padded tail of the final batch so it contributes nothing
                batch["masks"][tbs:, :] = False
                batch["is_targets"][tbs:, :] = False
                batch["is_padding"][tbs:, :] = True
                _, yhat = net(batch)
                is_t = batch["is_targets"].bool()
                if kind == "clf":
                    p = torch.sigmoid(yhat["boolean"][is_t].float()).flatten()
                    y = batch["boolean_values"][is_t].float().flatten()
                else:
                    p = yhat["number"][is_t].float().flatten()
                    y = batch["number_values"][is_t].float().flatten()
                preds.append(p.cpu())
                labels.append(y.cpu())
                n_seen += int(p.numel())
        if not preds:
            rec.update(status="no-batches", n=0)
            return rec

        p = torch.cat(preds).numpy()
        y = torch.cat(labels).numpy()
        rec["n"] = int(len(y))
        if kind == "clf":
            yb = (y > 0).astype(int)
            if yb.min() == yb.max():
                rec.update(status="single-class", auroc=float("nan"))
            else:
                rec.update(status="ok", auroc=float(roc_auc_score(yb, p)),
                           pos_rate=float(yb.mean()))
        else:
            # values are z-scored per column by the preprocessor; R2 is scale-invariant so it is
            # comparable across arms, while MAE stays in normalized units
            rec.update(status="ok", r2=float(r2_score(y, p)),
                       mae_norm=float(mean_absolute_error(y, p)))
    except torch.cuda.OutOfMemoryError:
        rec.update(status="OOM", n=0)
    except RuntimeError as e:  # some paths surface OOM as a plain RuntimeError
        msg = str(e)
        if "out of memory" in msg.lower():
            rec.update(status="OOM", n=0)
        else:
            rec.update(status=f"RuntimeError: {msg[:200]}", n=0)
    except Exception as e:  # noqa: BLE001 - one bad cell must not kill the sweep
        rec.update(status=f"{type(e).__name__}: {str(e)[:200]}", n=0)
    finally:
        # Release the workers and any half-built batch before the caller retries at a smaller
        # batch -- otherwise the retry inherits this attempt's memory and OOMs again at every size.
        try:
            del loader
        except NameError:
            pass
        if device == "cuda":
            torch.cuda.empty_cache()
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def _free_gpu() -> None:
    """Release the previous checkpoint's memory. No-op without a CUDA torch."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_benchmark(
    checkpoints: dict[str, str | Path],
    *,
    tasks: list[tuple] | None = None,
    ctx_lens: tuple[int, ...] = CTX_LENS,
    out_csv: str | Path | None = None,
    device: str = "cuda",
    max_samples: int = 2048,
    arch: dict | None = None,
    resume: bool = True,
    retry_failed: bool = True,
    lean_masks: bool = False,
    **kw,
):
    """Sweep every (generator checkpoint, task, context length). Returns a DataFrame.

    Results are written after every row, so a session that dies mid-sweep keeps what it measured.
    """
    import pandas as pd

    tasks = tasks or tasks_for()
    rows: list[dict] = []
    done_keys: set[tuple] = set()
    # Resume: a 30k sweep runs for many hours and will outlive a Colab session. Rows already in
    # the CSV are kept and their cells skipped, so a rerun continues instead of starting over.
    if resume and out_csv and Path(out_csv).exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done_keys = {
            (r["generator"], r["db"], r["task"], int(r["ctx_len"]))
            for r in rows
            if str(r.get("status")) == "ok" or not retry_failed
        }
        print(f"[bench] resuming: {len(rows)} rows already in {out_csv}, "
              f"{len(done_keys)} cells will be skipped")
    total = len(checkpoints) * len(tasks) * len(ctx_lens)
    print(f"[bench] {len(checkpoints)} checkpoints x {len(tasks)} tasks x {len(ctx_lens)} "
          f"context lengths = {total} evaluations")
    free = free_vram_gb()
    plan = {c: plan_batch_size(c, lean_masks=lean_masks, free_gb=free) for c in ctx_lens}
    print(f"[bench] masks: {'LEAN (mask_mod BlockMask)' if lean_masks else 'stock dense'} | "
          f"free VRAM: {'unknown' if free is None else f'{free:.1f} GB'}")
    print(f"[bench] planned batch size: "
          + ", ".join(f"ctx {c}->{b}" for c, b in plan.items()))
    rep = contamination_report(list(checkpoints), sorted({t[0] for t in tasks}))
    if not rep["clean"]:
        print(f"[bench]  !! contaminated pairs (generator trained on that database): "
              f"{rep['flagged']}")

    done = 0
    for gen, ckpt in checkpoints.items():
        pending = [(db, t, ctx) for db, t, *_ in tasks for ctx in ctx_lens
                   if (gen, db, t, int(ctx)) not in done_keys]
        if not pending:
            print(f"[bench] {gen}: all cells already done, skipping checkpoint load")
            done += len(tasks) * len(ctx_lens)
            continue
        net = load_model(ckpt, device=device, arch=arch)
        for db, task, target, kind, drops in tasks:
            for ctx in ctx_lens:
                done += 1
                if (gen, db, task, int(ctx)) in done_keys:
                    continue
                r = evaluate_task(net, db, task, target, kind, ctx_len=ctx, drops=drops,
                                  device=device, max_samples=max_samples,
                                  lean_masks=lean_masks, **kw)
                r["generator"] = gen
                r["contaminated"] = db in CONTAMINATED.get(gen, ())
                rows.append(r)
                metric = r.get("auroc", r.get("r2"))
                print(f"[bench] {done}/{total} {gen:8s} {db:11s} {task:16s} ctx={ctx:<6d} "
                      f"{r['status']:14s} "
                      f"{'' if metric is None else f'{"auroc" if kind == "clf" else "r2"}={metric:.4f}'}",
                      flush=True)
                if out_csv:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
        del net
        _free_gpu()
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[bench] wrote {out_csv}")
    return df


#: A result cell. Sharded runs are recombined on this key.
CELL_KEY = ("generator", "db", "task", "ctx_len")


def merge_results(pattern: str | Path, out_csv: str | Path | None = None) -> "object":
    """Recombine the CSVs written by sharded benchmark sessions into one frame.

    Splitting the sweep across sessions means several CSVs covering disjoint (or overlapping) cells.
    Duplicates are resolved by preferring `status == "ok"`, then the later row, so re-running a
    shard that previously OOMed upgrades the result rather than adding a second row for it.
    """
    from glob import glob

    import pandas as pd

    paths = sorted(glob(str(pattern))) if not isinstance(pattern, (list, tuple)) else list(pattern)
    paths = [p for p in paths if not str(p).endswith("_summary.csv")]
    if not paths:
        raise FileNotFoundError(f"no result CSVs match {pattern}")
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        if len(d):
            frames.append(d)
        print(f"[merge] {Path(p).name:44s} {len(d):5d} rows")
    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df["_ok"] = (df["status"] == "ok").astype(int)
    df = (df.sort_values(["_ok"], kind="stable")
            .drop_duplicates(subset=list(CELL_KEY), keep="last")
            .drop(columns="_ok")
            .sort_values(list(CELL_KEY))
            .reset_index(drop=True))
    print(f"[merge] {before} rows from {len(paths)} shards -> {len(df)} unique cells")

    expected = len(set(df["generator"])) * len(set(zip(df["db"], df["task"]))) * len(
        set(df["ctx_len"]))
    if len(df) < expected:
        print(f"[merge]  !! {expected - len(df)} cells still missing "
              f"({len(df)}/{expected}) -- some shard has not run yet")
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[merge] wrote {out_csv}")
    return df


def summarize(df) -> "object":
    """Mean AUROC (classification) and R2 (regression) per generator x context length."""
    import pandas as pd

    ok = df[df["status"] == "ok"]
    out = []
    for kind, col in (("clf", "auroc"), ("reg", "r2")):
        sub = ok[(ok["kind"] == kind) & (~ok["contaminated"])]
        if sub.empty or col not in sub:
            continue
        piv = sub.pivot_table(index="generator", columns="ctx_len", values=col, aggfunc="mean")
        piv.columns = [f"{col}@{c}" for c in piv.columns]
        out.append(piv)
    return pd.concat(out, axis=1) if out else pd.DataFrame()
