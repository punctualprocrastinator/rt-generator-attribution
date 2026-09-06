"""RT pretraining driver: paper architecture, wall-clock-calibrated step budget.

Training itself is RT's own `rt.main.main()` -- we do not reimplement the loop, so the objective
(masked-cell prediction), the relational attention masks, the optimizer (AdamW, wd 0.1) and the
schedule (OneCycleLR, 20% linear warmup then linear decay, i.e. the paper's schedule) are upstream
code. What this module adds is the experiment scaffolding:

  * the paper's architecture and context length, pinned as constants;
  * a wall-clock calibration so each generator arm gets the SAME number of optimizer steps within
    a fixed time budget, and that step count is shared across arms via one JSON on Drive
    (otherwise "3 hours on whatever GPU Colab handed out" would silently differ per arm and
    confound the comparison);
  * a final-checkpoint copy with a config.json, because `rt.main` names checkpoints
    `steps=N.pt` and its `*_best.pt` selection compares a LOSS with `>` (higher-is-better),
    which picks the worst checkpoint when the only eval metric is a synthetic-DB loss.

In-context learning
-------------------
Nothing extra is needed to "turn on" ICL: RT is pretrained with masked-cell prediction where the
ONLY masked cell is the target cell of the seed row (`masks == is_targets` in the Rust sampler).
Every other cell in the BFS-sampled context stays visible -- including the SAME column in sibling
rows reached through shared parents -- and the per-block *column* attention lets the masked cell
attend to exactly those. That is the in-context supply of labelled examples, which is why the
pretrained model can be prompted zero-shot on an unseen schema. Our job is to keep `ctx_len` at the
paper's 1024 and `max_bfs_width` wide enough that sibling rows actually land in the window.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

# RT paper Sec 4.1 "Architecture details": 12 layers, d_model 256, 8 heads, gated MLP d_ff 1024,
# MiniLM (384-dim) text embeddings -> ~22M parameters.
PAPER_ARCH = dict(
    embedding_model="all-MiniLM-L12-v2",
    d_text=384,
    num_blocks=12,
    d_model=256,
    num_heads=8,
    d_ff=1024,
)
# Paper pretrains at context length 1024. max_bfs_width is the P->F subsample bound; 128 is the
# value PluRel's synthetic pretraining script uses.
PAPER_CTX = dict(ctx_len=1024, max_bfs_width=128)

# Paper: AdamW, weight decay 0.1, peak lr 1e-3 at batch 256, 20% linear warmup, linear decay.
# PluRel's synthetic-pretraining script runs batch 128 at lr 5e-4 -- i.e. lr scaled linearly with
# batch size. We keep that rule so a smaller batch (needed for a single-GPU time budget) stays
# consistent with both references.
LR_REFERENCE = dict(lr=5e-4, batch_size=128)
PAPER_OPT = dict(wd=0.1, lr_schedule=True, max_grad_norm=1.0)


def scaled_lr(batch_size: int, reference: dict | None = None) -> float:
    r = reference or LR_REFERENCE
    return r["lr"] * batch_size / r["batch_size"]


# --------------------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------------------- #
def calibrate_sec_per_step(
    tasks: list[tuple],
    *,
    batch_size: int,
    num_workers: int = 2,
    measure_steps: int = 20,
    warmup_steps: int = 6,
    compile_: bool = True,
    max_tasks: int = 64,
    seed: int = 0,
) -> dict:
    """Time forward+backward+step on real batches. Returns {sec_per_step, ...}.

    Uses a subset of the task list: throughput is set by `ctx_len` x `batch_size` and the model,
    not by how many tasks feed the sampler, and a subset keeps setup to seconds.
    """
    import torch
    from torch import optim
    from torch.utils.data import DataLoader

    from rt.data import RelationalDataset
    from rt.model import RelationalTransformer

    sub = tasks[:max_tasks]
    ds = RelationalDataset(
        tasks=[(db, tbl, col, "train", drop) for db, tbl, col, drop in sub],
        batch_size=batch_size,
        rank=0,
        world_size=1,
        ctx_len=PAPER_CTX["ctx_len"],
        max_bfs_width=PAPER_CTX["max_bfs_width"],
        embedding_model=PAPER_ARCH["embedding_model"],
        d_text=PAPER_ARCH["d_text"],
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=True, in_order=True)

    net = RelationalTransformer(
        num_blocks=PAPER_ARCH["num_blocks"],
        d_model=PAPER_ARCH["d_model"],
        d_text=PAPER_ARCH["d_text"],
        num_heads=PAPER_ARCH["num_heads"],
        d_ff=PAPER_ARCH["d_ff"],
    )
    n_params = sum(p.numel() for p in net.parameters())
    net = net.to("cuda").to(torch.bfloat16)
    opt = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.1, fused=True)
    netc = torch.compile(net, dynamic=False, disable=not compile_)

    it = iter(loader)
    t_start = None
    n = 0
    for i in range(warmup_steps + measure_steps):
        try:
            batch = next(it)
        except StopIteration:  # tiny task list -> loader shorter than the measurement window
            it = iter(loader)
            batch = next(it)
        batch.pop("true_batch_size", None)
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        loss, _ = netc(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == warmup_steps - 1:
            torch.cuda.synchronize()
            t_start = time.time()
        elif i >= warmup_steps:
            n += 1
    torch.cuda.synchronize()
    sec = (time.time() - t_start) / max(n, 1)

    del netc, net, opt, loader, ds
    torch.cuda.empty_cache()
    return {
        "sec_per_step": sec,
        "batch_size": batch_size,
        "ctx_len": PAPER_CTX["ctx_len"],
        "param_count": n_params,
        "measured_steps": n,
    }


def resolve_max_steps(
    *,
    budget_hours: float,
    sec_per_step: float,
    shared_budget_path: str | Path | None = None,
    generator: str = "",
    gpu: str = "",
    overhead_frac: float = 0.12,
    round_to: int = 500,
) -> dict:
    """Pick the step budget, shared across generator arms.

    The FIRST arm to run computes `max_steps` from its measured throughput and writes it to
    `shared_budget_path`; later arms reuse that exact number so every arm sees the same amount of
    data. If a later arm's own throughput implies a very different wall clock (different GPU class),
    we keep the shared step count -- comparability wins -- and print a loud warning with the
    projected time so the discrepancy is visible rather than silent.

    `overhead_frac` reserves time for evals, checkpoint writes and dataloader stalls.
    """
    usable = budget_hours * 3600 * (1.0 - overhead_frac)
    # `+ 1` is deliberate and load-bearing: rt.main checks `steps % eval_freq == 0` at the TOP of
    # the loop and stops at `steps == max_steps`, so an N+1 step budget with eval_freq dividing N
    # puts the last evaluation -- and therefore the last checkpoint -- at step N, after all the
    # training. A plain multiple of eval_freq would leave the final chunk of training unsaved.
    # (PluRel's own scripts use 4_001 / 8_001 / ... for the same reason.)
    own = max(round_to, int(usable / max(sec_per_step, 1e-9) // round_to * round_to)) + 1

    out = {
        "max_steps": own,
        "source": "measured",
        "own_projection_hours": own * sec_per_step / 3600,
        "sec_per_step": sec_per_step,
        "generator": generator,
        "gpu": gpu,
        "budget_hours": budget_hours,
    }
    if shared_budget_path is None:
        return out

    p = Path(shared_budget_path)
    if p.exists():
        shared = json.loads(p.read_text())
        steps = int(shared["max_steps"])
        proj = steps * sec_per_step / 3600
        out.update(
            max_steps=steps,
            source=f"shared (set by {shared.get('generator', '?')} on {shared.get('gpu', '?')})",
            own_projection_hours=proj,
            shared=shared,
        )
        if proj > budget_hours * 1.5 or proj < budget_hours * 0.5:
            print(
                f"\n  !! WALL-CLOCK MISMATCH: the shared budget is {steps:,} steps, which on THIS "
                f"GPU ({gpu}) projects to {proj:.1f} h, not the {budget_hours:.1f} h target.\n"
                f"     Keeping {steps:,} steps so all generator arms train on equal data.\n"
                f"     Set SHARED_STEP_BUDGET=False to opt out, or rerun the other arms on the "
                f"same GPU class.\n"
            )
        return out

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"  wrote shared step budget -> {p}  ({own:,} steps)")
    return out


# --------------------------------------------------------------------------------------- #
# pretraining
# --------------------------------------------------------------------------------------- #
def patch_main_checkpointing(repo: str | Path) -> bool:
    """Make `rt.main` checkpoint on schedule even when there are no eval metrics.

    Upstream only saves inside `for (db, table), metric in metrics["val"].items():`, so a run with
    an empty `eval_tasks` list -- which is what pure pretraining wants, since these synthetic
    databases are training data and not a benchmark -- would train for hours and save nothing.
    """
    p = Path(repo) / "rt" / "main.py"
    src = p.read_text(encoding="utf-8")
    marker = (
        '                if save_ckpt_dir is not None:\n'
        '                    for (db_name, table_name), metric in metrics["val"].items():'
    )
    patched = (
        '                if save_ckpt_dir is not None:\n'
        '                    if not metrics["val"]:\n'
        '                        checkpoint()\n'
        '                    for (db_name, table_name), metric in metrics["val"].items():'
    )
    if patched in src:
        return False
    if marker not in src:
        raise RuntimeError(
            f"could not find the checkpoint block in {p} -- rt/main.py changed upstream; update "
            f"rt_icl.train.patch_main_checkpointing"
        )
    p.write_text(src.replace(marker, patched), encoding="utf-8")
    import ast

    ast.parse(p.read_text(encoding="utf-8"))
    print(f"[train] patched {p}: checkpoint on schedule when there are no eval metrics")
    return True


def pretrain(
    *,
    generator: str,
    train_tasks: list[tuple],
    eval_tasks: list[tuple],
    save_ckpt_dir: str | Path,
    batch_size: int,
    max_steps: int,
    lr: float | None = None,
    num_workers: int = 2,
    eval_batch_size: int = 128,
    n_evals: int = 4,
    max_eval_steps: int = 1,
    compile_: bool = True,
    seed: int = 0,
    project: str = "rt-rdg-study",
    wandb_mode: str = "offline",
    repo: str | Path | None = None,
    extra_config: dict | None = None,
) -> dict:
    """Run RT pretraining via `rt.main.main`, then materialize a clean final checkpoint."""
    os.environ.setdefault("WANDB_MODE", wandb_mode)
    os.environ.setdefault("WANDB_SILENT", "true")

    if not eval_tasks:
        # pure pretraining: no benchmark here, so checkpointing must not depend on eval metrics
        patch_main_checkpointing(repo or Path(__file__).resolve().parents[2] / "plurel")

    from rt.main import main

    save_dir = Path(save_ckpt_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    lr = lr if lr is not None else scaled_lr(batch_size)

    # Pick eval_freq so that (max_steps - 1) is an exact multiple of it: that guarantees an
    # evaluation -- and hence a checkpoint -- at step max_steps-1, i.e. after all the training.
    # max_steps is snapped down to fit, so the schedule and the checkpoint stay consistent.
    eval_freq = max(1, (max_steps - 1) // max(n_evals, 1))
    max_steps = eval_freq * max(n_evals, 1) + 1

    cfg = dict(
        generator=generator,
        n_train_tasks=len(train_tasks),
        n_eval_tasks=len(eval_tasks),
        batch_size=batch_size,
        max_steps=max_steps,
        lr=lr,
        eval_freq=eval_freq,
        seed=seed,
        **PAPER_ARCH,
        **PAPER_CTX,
        **PAPER_OPT,
    )
    if extra_config:
        cfg.update(extra_config)
    (save_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(json.dumps(cfg, indent=2, default=str))

    t0 = time.time()
    main(
        project=project,
        # synthetic databases have no separate val/test task splits, so eval is the held-out-DB
        # masked-cell loss on the 'val' split only (rt.main asserts exactly one synthetic split)
        eval_splits=["val"],
        eval_freq=eval_freq,
        eval_pow2=False,
        max_eval_steps=max_eval_steps,
        load_ckpt_path=None,
        save_ckpt_dir=str(save_dir),
        compile_=compile_,
        seed=seed,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        lr=lr,
        max_steps=max_steps,
        **PAPER_CTX,
        **PAPER_OPT,
        **PAPER_ARCH,
    )
    elapsed = time.time() - t0

    final = finalize_checkpoint(save_dir, generator=generator, cfg=cfg, elapsed_s=elapsed)
    print(f"\ntrained {max_steps:,} steps in {elapsed/3600:.2f} h -> {final}")
    return {"final_checkpoint": str(final), "elapsed_hours": elapsed / 3600, "config": cfg}


def finalize_checkpoint(
    save_dir: str | Path, *, generator: str, cfg: dict, elapsed_s: float | None = None
) -> Path:
    """Copy the highest-step `steps=N.pt` to `<generator>_final.pt` with its config alongside.

    We deliberately ignore `*_best.pt`: `rt.main` selects "best" with `metric > best_metric`, but
    for synthetic-only evaluation the metric IS a loss, so its "best" is the worst checkpoint.
    """
    save_dir = Path(save_dir)
    ckpts = []
    for p in save_dir.glob("steps=*.pt"):
        try:
            ckpts.append((int(p.stem.split("=")[1]), p))
        except (IndexError, ValueError):
            continue
    if ckpts:
        steps, src = max(ckpts)
    else:
        # rt.main writes `<db>_<table>_best.pt` instead of `steps=N.pt` on an "improving" eval, so a
        # very short run can produce only those. Fall back to the most recently written checkpoint.
        alt = sorted(save_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not alt:
            raise FileNotFoundError(
                f"no checkpoint in {save_dir} -- rt.main only checkpoints at eval steps, so check "
                f"that eval_freq <= max_steps"
            )
        src, steps = alt[-1], -1
    dst = save_dir / f"{generator}_final.pt"
    shutil.copy(src, dst)
    meta = dict(cfg)
    meta.update(final_from=src.name, final_steps=steps, elapsed_s=elapsed_s)
    (save_dir / f"{generator}_final.json").write_text(json.dumps(meta, indent=2, default=str))
    return dst


def smoke_train(train_tasks, eval_tasks, save_ckpt_dir, *, batch_size=8, steps=6, **kw) -> dict:
    """Tiny end-to-end run: proves data -> sampler -> model -> loss -> checkpoint all work."""
    return pretrain(
        generator="smoke",
        train_tasks=train_tasks,
        eval_tasks=eval_tasks[:2],
        save_ckpt_dir=save_ckpt_dir,
        batch_size=batch_size,
        max_steps=steps,
        num_workers=0,
        eval_batch_size=10,
        n_evals=1,
        compile_=False,
        **kw,
    )
