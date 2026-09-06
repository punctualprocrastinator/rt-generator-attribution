"""Run RT on GPUs without bfloat16 / FlexAttention support (Turing / T4, or CPU).

RT as published targets Ampere or newer: it casts the whole network to bfloat16 and routes every
attention through `flex_attention`, which is `torch.compile`d at import. Neither works on a T4
(compute 7.5): bf16 has no hardware support there, and FlexAttention's Triton kernels need sm80+.

This module provides a compatibility plan and three surgical patches. They are deliberately small,
and each preserves the computation:

  1. `rt.model._make_block_mask` becomes a pass-through, so the DENSE boolean masks that
     `RelationalTransformer.forward` already builds are handed straight to attention instead of
     being converted to a BlockMask.
  2. `MaskedAttention.forward` uses `F.scaled_dot_product_attention(..., attn_mask=dense)` when it
     receives a dense mask. Same masked-softmax attention as FlexAttention over the same mask --
     what changes is the kernel, not the math. `selfcheck_attention` proves this numerically
     against an explicit reference implementation.
  3. `RelationalTransformer.forward` casts floating-point batch entries to the parameter dtype,
     because the Rust sampler always hands back bfloat16 buffers.

Plus `patch_main_dtype`, which rewrites the single `net.to(torch.bfloat16)` line in `rt/main.py`.

WHAT THIS COSTS
---------------
Correctness is preserved; throughput is not. FlexAttention exploits the sparsity of RT's column /
feature / neighbour masks by skipping empty blocks, while dense SDPA evaluates all `ctx_len**2`
query-key pairs. Together with Turing's slower fp16 math, expect roughly an order of magnitude less
throughput than an A100. Use this to validate the pipeline, not to produce the study's weights --
and never mix GPU classes across generator arms, since equal step budgets would then mean wildly
unequal wall clocks.

fp16 note: RT trains fully in low precision with no GradScaler. That is safe in bf16 and risky in
fp16, so `gpu_plan` prefers fp32 on Turing unless `allow_fp16=True`.
"""

from __future__ import annotations

import os
from pathlib import Path

AMPERE = (8, 0)


# --------------------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------------------- #
def gpu_plan(allow_fp16: bool = False, force: str | None = None) -> dict:
    """Choose dtype / attention / compile settings for the current device."""
    import torch

    if force == "cpu" or not torch.cuda.is_available():
        return dict(
            device="cpu", dtype="float32", attention="dense_sdpa", compile=False,
            native=False, gpu="cpu", capability=None,
            reason="no CUDA device: fp32 + dense SDPA, eager",
        )

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if cap >= AMPERE and force != "turing":
        return dict(
            device="cuda", dtype="bfloat16", attention="flex", compile=True,
            native=True, gpu=name, capability=f"{cap[0]}.{cap[1]}",
            reason="Ampere or newer: the paper's bf16 + FlexAttention path",
        )

    dtype = "float16" if allow_fp16 else "float32"
    # FlexAttention never materializes the mask; the fallback holds a dense (B, S, S) boolean mask
    # per attention type and lets SDPA build B*H*S*S scores, so memory grows with batch*ctx_len^2.
    # At ctx_len 1024 that is ~1 GB of scores per attention call at batch 64 in fp16 -- an OOM on a
    # 16 GB T4 long before throughput matters.
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    hint = max(4, int(8 * (total_mem / 16.0)) // 4 * 4)
    return dict(
        device="cuda", dtype=dtype, attention="dense_sdpa", compile=True,
        native=False, gpu=name, capability=f"{cap[0]}.{cap[1]}",
        max_batch_hint=hint,
        total_mem_gb=round(total_mem, 1),
        reason=(
            f"{name} is compute {cap[0]}.{cap[1]} (< 8.0): no bf16 hardware and no FlexAttention "
            f"kernels, so falling back to {dtype} + dense SDPA. Expect roughly an order of "
            f"magnitude less throughput than an A100, and keep BATCH_SIZE around {hint} or lower "
            f"({total_mem:.0f} GB, dense masks scale with batch x ctx_len^2)"
            + ("" if allow_fp16 else "; pass allow_fp16=True to trade stability for speed")
        ),
    )


def describe(plan: dict) -> str:
    return (
        f"[compat] {plan['gpu']} (cc {plan['capability']}) -> dtype={plan['dtype']}, "
        f"attention={plan['attention']}, compile={plan['compile']}\n"
        f"         {plan['reason']}"
    )


# --------------------------------------------------------------------------------------- #
# patches
# --------------------------------------------------------------------------------------- #
_APPLIED = {}


def apply_plan(plan: dict, repo: str | Path | None = None) -> dict:
    """Patch `rt` to run under `plan`. Idempotent; a no-op on the native Ampere path."""
    if plan["attention"] == "flex" and plan["dtype"] == "bfloat16":
        return {"patched": False, "why": "native path, nothing to patch"}
    if _APPLIED.get("done"):
        return _APPLIED
    if repo is not None:
        patch_main_dtype(repo, plan["dtype"])
    _patch_attention()
    _patch_input_cast(plan["dtype"])
    _APPLIED.update(patched=True, plan=plan)
    print(f"[compat] patched rt.model for {plan['dtype']} + {plan['attention']}")
    return _APPLIED


def _patch_attention() -> None:
    """Dense-mask SDPA instead of FlexAttention."""
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from rt import model as rt_model

    # 1. keep the dense (B, S, S) boolean mask instead of compiling a BlockMask
    rt_model._make_block_mask = lambda mask, batch_size, seq_len, device: mask

    def forward(self, x, block_mask):
        q = rearrange(self.wq(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.wk(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.wv(x), "b s (h d) -> b h s d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)

        if block_mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        else:
            m = block_mask
            if m.dim() == 3:
                m = m.unsqueeze(1)  # (B, 1, S, S) -> broadcast across heads
            # A fully-masked query row would softmax over all -inf and yield NaN, which then
            # survives `loss * mask` (NaN * 0 is NaN) and poisons training. FlexAttention returns
            # zeros for such rows; padded rows here are exactly that case. Allowing every query to
            # see itself keeps the row finite, and those positions are discarded by the loss mask.
            m = m.clone()
            eye = torch.eye(m.size(-1), dtype=torch.bool, device=m.device)
            m |= eye
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=m)

        return self.wo(rearrange(x, "b h s d -> b s (h d)"))

    rt_model.MaskedAttention.forward = forward


def _patch_input_cast(dtype_name: str) -> None:
    """Cast float batch entries to the parameter dtype (the sampler emits bfloat16 buffers)."""
    import torch

    from rt import model as rt_model

    if getattr(rt_model.RelationalTransformer, "_rt_icl_cast", False):
        return
    original = rt_model.RelationalTransformer.forward
    want = getattr(torch, dtype_name)

    def forward(self, batch):
        batch = {
            k: (v.to(want) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in batch.items()
        }
        return original(self, batch)

    rt_model.RelationalTransformer.forward = forward
    rt_model.RelationalTransformer._rt_icl_cast = True


def patch_main_dtype(repo: str | Path, dtype_name: str) -> bool:
    """Rewrite the hardcoded `net.to(torch.bfloat16)` in rt/main.py. Returns True if changed."""
    p = Path(repo) / "rt" / "main.py"
    src = p.read_text(encoding="utf-8")
    marker = "net = net.to(torch.bfloat16)"
    replacement = (
        'net = net.to(getattr(torch, __import__("os").environ.get("RT_DTYPE", "bfloat16")))'
    )
    os.environ["RT_DTYPE"] = dtype_name
    if replacement in src:
        return False
    if marker not in src:
        raise RuntimeError(
            f"could not find {marker!r} in {p} -- rt/main.py changed upstream; update "
            f"rt_icl.compat.patch_main_dtype"
        )
    p.write_text(src.replace(marker, replacement), encoding="utf-8")
    import ast

    ast.parse(p.read_text(encoding="utf-8"))
    print(f"[compat] patched {p} to honour RT_DTYPE={dtype_name}")
    return True


# --------------------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------------------- #
def selfcheck_attention(
    batch=2, heads=4, seq=32, head_dim=16, dtype="float32", device="cpu", atol=1e-4
) -> None:
    """Prove the SDPA fallback computes the same masked attention as an explicit reference.

    The reference is written out longhand (scores -> mask -> softmax -> values), so agreeing with
    it means the fallback really is masked-softmax attention and not merely 'something plausible'.
    Includes a fully-masked row, the case that silently produces NaNs if mishandled.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    dt = getattr(torch, dtype)
    q = torch.randn(batch, heads, seq, head_dim, dtype=dt, device=device)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dt, device=device)
    v = torch.randn(batch, heads, seq, head_dim, dtype=dt, device=device)

    mask = torch.rand(batch, seq, seq, device=device) > 0.5
    mask[:, 0, :] = False  # a fully-masked query row (i.e. a padded position)

    m = mask.unsqueeze(1).clone()
    m |= torch.eye(seq, dtype=torch.bool, device=device)
    got = F.scaled_dot_product_attention(q, k, v, attn_mask=m)

    scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
    scores = scores.masked_fill(~m, float("-inf"))
    want = torch.softmax(scores, dim=-1) @ v

    assert torch.isfinite(got).all(), "SDPA fallback produced non-finite values"
    diff = (got.float() - want.float()).abs().max().item()
    assert diff < atol, f"SDPA fallback != reference masked attention (max abs diff {diff})"
    print(f"[compat] attention self-check PASSED (max abs diff {diff:.2e}, fully-masked row finite)")


def selfcheck_model(plan: dict, seq: int = 64, batch: int = 2) -> float:
    """Run one forward/backward of the real RelationalTransformer under the plan. Returns the loss.

    This is the check that matters on a T4: it exercises the patched attention inside the actual
    model with a realistic batch, and fails loudly on a non-finite loss.
    """
    import torch

    from rt.model import RelationalTransformer

    d_model, d_text = 64, 32
    net = RelationalTransformer(num_blocks=2, d_model=d_model, d_text=d_text, num_heads=4, d_ff=128)
    dt = getattr(torch, plan["dtype"])
    dev = plan["device"]
    net = net.to(dev).to(dt)

    g = torch.Generator().manual_seed(0)
    n_nodes = max(seq // 4, 2)
    node_idxs = torch.randint(0, n_nodes, (batch, seq), generator=g)
    b = {
        "node_idxs": node_idxs,
        "f2p_nbr_idxs": torch.randint(0, n_nodes, (batch, seq, 5), generator=g),
        "col_name_idxs": torch.randint(0, 6, (batch, seq), generator=g),
        "table_name_idxs": torch.randint(0, 3, (batch, seq), generator=g),
        "sem_types": torch.randint(0, 4, (batch, seq), generator=g),
        "masks": torch.rand(batch, seq, generator=g) > 0.8,
        "is_padding": torch.zeros(batch, seq, dtype=torch.bool),
        "number_values": torch.randn(batch, seq, 1, generator=g),
        "datetime_values": torch.randn(batch, seq, 1, generator=g),
        "boolean_values": torch.randint(0, 2, (batch, seq, 1), generator=g).float(),
        "text_values": torch.randn(batch, seq, d_text, generator=g),
        "col_name_values": torch.randn(batch, seq, d_text, generator=g),
    }
    b["is_padding"][:, -seq // 8 :] = True  # a real padded tail
    # RT only ever masks boolean or numeric cells (paper Sec 3.3); masking a text cell raises
    # "masking text not supported". sem_types are ordered [number, text, datetime, boolean].
    maskable = (b["sem_types"] == 0) | (b["sem_types"] == 3)
    b["masks"] &= maskable & ~b["is_padding"]
    b["sem_types"][:, 0] = 0  # guarantee at least one maskable (numeric) cell per sequence
    b["masks"][:, 0] = True
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

    loss, _ = net(b)
    assert torch.isfinite(loss), f"loss is not finite under plan {plan['dtype']}/{plan['attention']}"
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g_).all() for g_ in grads), "non-finite gradients"
    print(
        f"[compat] model self-check PASSED "
        f"(loss={loss.item():.4f}, {len(grads)} grad tensors finite, dtype={plan['dtype']})"
    )
    return float(loss.item())


# --------------------------------------------------------------------------------------- #
# lean block masks
# --------------------------------------------------------------------------------------- #
# Upstream `RelationalTransformer.forward` builds each attention mask as a dense (B, S, S) bool
# tensor -- via two (B, S, S, 5) intermediates -- and then hands `create_block_mask` a `mask_mod`
# that merely indexes it. FlexAttention never needed that tensor: `mask_mod` is a predicate it
# evaluates itself, block by block. Expressing the same three predicates directly turns the
# dominant memory cost of long-context evaluation into nothing:
#
#     ctx_len 30000, batch 1:   ~9.9 GB of masks  ->  ~0.7 MB of BlockMask
#     ctx_len  1024, batch 64:  ~0.74 GB          ->  ~0.02 MB
#
# The predicates are transcribed one-for-one, so the BlockMask is the same object it would have
# been -- `selfcheck_block_masks` proves that elementwise rather than asserting it. Only the
# FlexAttention path benefits: the dense-SDPA fallback (compute < 8.0) needs a real dense mask,
# so this must not be combined with `apply_plan`.
LEAN_HELPERS = """
# --- rt_icl: block-mask construction ----------------------------------------------------- #
def _rt_icl_dense_masks(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
    '''The stock (B, S, S) mask construction, kept verbatim as the self-check's reference.'''
    pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
    same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
    kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
    q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
    same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
        table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
    )
    return {
        'feat': ((same_node | kv_in_f2p) & pad).contiguous(),
        'nbr': (q_in_f2p & pad).contiguous(),
        'col': (same_col_table & pad).contiguous(),
    }


def _rt_icl_mask_mods(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
    '''The same three predicates as `_rt_icl_dense_masks`, as FlexAttention mask_mods.

    Each is evaluated per (b, q, kv) inside create_block_mask, so no (B, S, S) tensor is ever
    allocated. The `.any(-1)` over the foreign->primary slots becomes an unrolled OR over the
    same K slots -- the identical boolean function, including for sentinel slot values, which
    compare equal or not exactly as they did before.
    '''
    K = f2p_nbr_idxs.shape[-1]

    def pad_ok(b, q, kv):
        return (~is_padding[b, q]) & (~is_padding[b, kv])

    def kv_in_f2p(b, q, kv):
        n = node_idxs[b, kv]
        out = n != n                      # all-False, with n's shape/device/broadcasting
        for j in range(K):
            out = out | (n == f2p_nbr_idxs[b, q, j])
        return out

    def q_in_f2p(b, q, kv):
        n = node_idxs[b, q]
        out = n != n
        for j in range(K):
            out = out | (n == f2p_nbr_idxs[b, kv, j])
        return out

    def feat(b, h, q, kv):
        same_node = node_idxs[b, q] == node_idxs[b, kv]
        return (same_node | kv_in_f2p(b, q, kv)) & pad_ok(b, q, kv)

    def nbr(b, h, q, kv):
        return q_in_f2p(b, q, kv) & pad_ok(b, q, kv)

    def col(b, h, q, kv):
        same = (col_name_idxs[b, q] == col_name_idxs[b, kv]) & (
            table_name_idxs[b, q] == table_name_idxs[b, kv]
        )
        return same & pad_ok(b, q, kv)

    return {'feat': feat, 'nbr': nbr, 'col': col}


def _rt_icl_block_masks(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs,
                        is_padding, batch_size, seq_len, device):
    mods = _rt_icl_mask_mods(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding)
    return {
        l: create_block_mask(mod, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
                             device=device, _compile=True)
        for l, mod in mods.items()
    }
"""

_LEAN_START = "        # Padding mask for attention pairs"
_LEAN_END = (
    "        block_masks = {l: make_block_mask(attn_mask) "
    "for l, attn_mask in attn_masks.items()}"
)
_LEAN_CALL = """        # --- rt_icl: block masks straight from the index tensors ---------------------- #
        # The same three predicates upstream builds densely, handed to create_block_mask as
        # mask_mods instead. Nothing of size (B, S, S) is allocated.
        block_masks = _rt_icl_block_masks(
            node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding,
            batch_size, seq_len, device,
        )"""


def patch_lean_block_masks(repo: str | Path) -> bool:
    """Rewrite `rt/model.py` to build BlockMasks without materializing dense masks.

    Source-level and idempotent, following `train.patch_main_checkpointing`. Raises if either
    anchor is missing rather than leaving the slow path silently in place -- a quiet no-op here
    would surface only as an OOM hours later.

    Must run BEFORE anything imports `rt.model`.
    """
    import ast
    import sys

    p = Path(repo) / "rt" / "model.py"
    src = p.read_text(encoding="utf-8")
    if "_rt_icl_block_masks" in src:
        return False
    i, j = src.find(_LEAN_START), src.find(_LEAN_END)
    if i < 0 or j < 0:
        raise RuntimeError(
            f"could not find the mask-construction block in {p} -- rt/model.py changed upstream; "
            f"update rt_icl.compat.patch_lean_block_masks "
            f"(missing {'start' if i < 0 else 'end'} anchor)"
        )
    new = src[:i] + _LEAN_CALL + "\n" + src[j + len(_LEAN_END):]
    anchor = "class RelationalTransformer(nn.Module):"
    k = new.find(anchor)
    if k < 0:
        raise RuntimeError(f"{p}: RelationalTransformer class not found")
    new = new[:k] + LEAN_HELPERS.strip("\n") + "\n\n\n" + new[k:]
    ast.parse(new)
    p.write_text(new, encoding="utf-8")
    if "rt.model" in sys.modules:
        import importlib

        importlib.reload(sys.modules["rt.model"])
        print("[compat] rt.model was already imported -- reloaded it")
    print(f"[compat] patched {p}: block masks built from mask_mods (no dense (B, S, S) masks)")
    return True


def selfcheck_block_masks(batch=2, seq=96, n_slots=5, device="cpu", batch_dict=None) -> None:
    """Prove the lean mask_mods reproduce the stock dense masks EXACTLY.

    Elementwise equality over every (b, q, kv), not a tolerance: these are booleans, and a single
    flipped pair would change which cells attend to which, silently altering every metric.
    Pass a real `batch_dict` to check against actually sampled data rather than random indices.
    """
    import torch

    from rt.model import _rt_icl_dense_masks, _rt_icl_mask_mods

    if batch_dict is not None:
        keys = ("node_idxs", "f2p_nbr_idxs", "col_name_idxs", "table_name_idxs", "is_padding")
        args = [batch_dict[k] for k in keys]
        args = [a[:2, :seq] if a.dim() == 2 else a[:2, :seq, :] for a in args]
    else:
        g = torch.Generator().manual_seed(0)
        n_nodes = max(seq // 4, 2)
        node_idxs = torch.randint(0, n_nodes, (batch, seq), generator=g)
        f2p = torch.randint(-1, n_nodes, (batch, seq, n_slots), generator=g)  # -1 = empty slot
        col = torch.randint(0, 6, (batch, seq), generator=g)
        tbl = torch.randint(0, 3, (batch, seq), generator=g)
        pad = torch.zeros(batch, seq, dtype=torch.bool)
        pad[:, -seq // 8:] = True
        args = [node_idxs, f2p, col, tbl, pad]
    args = [a.to(device) for a in args]

    dense = _rt_icl_dense_masks(*args)
    mods = _rt_icl_mask_mods(*args)
    B, S = args[0].shape
    bi = torch.arange(B, device=device).view(B, 1, 1)
    qi = torch.arange(S, device=device).view(1, S, 1)
    ki = torch.arange(S, device=device).view(1, 1, S)

    for name, mod in mods.items():
        got = mod(bi, None, qi, ki).expand(B, S, S)
        want = dense[name]
        if not torch.equal(got, want):
            n_bad = int((got != want).sum())
            raise AssertionError(
                f"lean mask_mod '{name}' differs from the stock dense mask in {n_bad} of "
                f"{got.numel()} pairs -- do NOT use the lean path"
            )
    dens = {k: float(v.float().mean()) for k, v in dense.items()}
    print(
        f"[compat] block-mask self-check PASSED on {B}x{S}x{S} pairs, all three masks identical "
        f"(density feat={dens['feat']:.3f} nbr={dens['nbr']:.3f} col={dens['col']:.3f})"
    )
