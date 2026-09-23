# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact block64 Q-major dQ kernel used by the split BSA backward path.

One Triton program owns one physical Q64 block and traverses the active prefix
of its q2k row.  dQ stays in FP32 registers for the entire row and is written
once, avoiding the fused K-major kernel's per-edge FP32 global reduction.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - exercised only in CPU-only installs
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _bsa_dq_qmajor_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        neg_delta_ptr,
        q2k_ptr,
        q2k_num_ptr,
        block_sizes_ptr,
        dq_ptr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qs: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_dob: tl.constexpr,
        stride_doh: tl.constexpr,
        stride_dos: tl.constexpr,
        stride_dod: tl.constexpr,
        stride_lseb: tl.constexpr,
        stride_lseh: tl.constexpr,
        stride_lses: tl.constexpr,
        stride_db: tl.constexpr,
        stride_dh: tl.constexpr,
        stride_ds: tl.constexpr,
        stride_q2kb: tl.constexpr,
        stride_q2kh: tl.constexpr,
        stride_q2kq: tl.constexpr,
        stride_q2kn: tl.constexpr,
        stride_qnb: tl.constexpr,
        stride_qnh: tl.constexpr,
        stride_qnq: tl.constexpr,
        stride_bsb: tl.constexpr,
        stride_bsk: tl.constexpr,
        stride_dqb: tl.constexpr,
        stride_dqh: tl.constexpr,
        stride_dqs: tl.constexpr,
        stride_dqd: tl.constexpr,
        seqlen_q,
        seqlen_k,
        num_heads,
        num_q_blocks,
        block_sparse_num,
        softmax_scale,
        HAS_VARIABLE_COUNTS: tl.constexpr,
        HAS_BLOCK_SIZES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        work_idx = tl.program_id(0)
        q_block = work_idx % num_q_blocks
        head_batch = work_idx // num_q_blocks
        head = head_batch % num_heads
        batch = head_batch // num_heads

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        q_token = q_block * BLOCK_M + offs_m
        q_valid = q_token < seqlen_q

        q_offsets = (
            batch * stride_qb
            + head * stride_qh
            + q_token[:, None] * stride_qs
            + offs_d[None, :] * stride_qd
        )
        do_offsets = (
            batch * stride_dob
            + head * stride_doh
            + q_token[:, None] * stride_dos
            + offs_d[None, :] * stride_dod
        )
        q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
        do = tl.load(do_ptr + do_offsets, mask=q_valid[:, None], other=0.0)
        lse = tl.load(
            lse_ptr
            + batch * stride_lseb
            + head * stride_lseh
            + q_token * stride_lses,
            mask=q_valid,
            other=0.0,
        )
        neg_delta = tl.load(
            neg_delta_ptr
            + batch * stride_db
            + head * stride_dh
            + q_token * stride_ds,
            mask=q_valid,
            other=0.0,
        )

        active_blocks = block_sparse_num
        if HAS_VARIABLE_COUNTS:
            active_blocks = tl.load(
                q2k_num_ptr
                + batch * stride_qnb
                + head * stride_qnh
                + q_block * stride_qnq
            )

        dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        log2_e = 1.4426950408889634
        score_scale = softmax_scale * log2_e
        lse_log2 = lse * log2_e
        for slot in tl.range(0, active_blocks, 1):
            k_block = tl.load(
                q2k_ptr
                + batch * stride_q2kb
                + head * stride_q2kh
                + q_block * stride_q2kq
                + slot * stride_q2kn
            )
            valid_k_count = BLOCK_M
            if HAS_BLOCK_SIZES:
                valid_k_count = tl.load(
                    block_sizes_ptr
                    + batch * stride_bsb
                    + k_block * stride_bsk
                )

            for n_base in tl.static_range(0, BLOCK_M, BLOCK_N):
                n_in_block = n_base + offs_n
                k_token = k_block * BLOCK_M + n_in_block
                k_valid = (n_in_block < valid_k_count) & (k_token < seqlen_k)
                k_offsets = (
                    batch * stride_kb
                    + head * stride_kh
                    + k_token[:, None] * stride_ks
                    + offs_d[None, :] * stride_kd
                )
                v_offsets = (
                    batch * stride_vb
                    + head * stride_vh
                    + k_token[:, None] * stride_vs
                    + offs_d[None, :] * stride_vd
                )
                k_tile = tl.load(
                    k_ptr + k_offsets,
                    mask=k_valid[:, None],
                    other=0.0,
                )
                v_tile = tl.load(
                    v_ptr + v_offsets,
                    mask=k_valid[:, None],
                    other=0.0,
                )

                valid_probability = q_valid[:, None] & k_valid[None, :]
                score = tl.dot(q, tl.trans(k_tile)) * score_scale
                p = tl.where(
                    valid_probability,
                    tl.math.exp2(score - lse_log2[:, None]),
                    0.0,
                )
                dp = tl.dot(do, tl.trans(v_tile))
                ds = (p * (dp + neg_delta[:, None])).to(tl.bfloat16)
                dq_acc += tl.dot(ds, k_tile)

        dq_offsets = (
            batch * stride_dqb
            + head * stride_dqh
            + q_token[:, None] * stride_dqs
            + offs_d[None, :] * stride_dqd
        )
        tl.store(
            dq_ptr + dq_offsets,
            (dq_acc * softmax_scale).to(tl.bfloat16),
            mask=q_valid[:, None],
        )


def qmajor_dq_available() -> bool:
    """Return whether the optional Triton implementation can be launched."""

    return triton is not None


def bsa_dq_qmajor_triton(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    neg_delta: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    *,
    block_sizes: Optional[torch.Tensor],
    q2k_block_nums: Optional[torch.Tensor],
    softmax_scale: float,
    dq: torch.Tensor,
    block_n: int = 32,
) -> torch.Tensor:
    """Run exact Q-major dQ on canonical BHSD tensors."""

    if triton is None:
        raise RuntimeError(
            "The split BSA backward backend requires Triton from the PyTorch "
            "CUDA environment"
        ) from _TRITON_IMPORT_ERROR
    if torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
        raise RuntimeError("Q-major split dQ currently requires SM100 or SM103")
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError("Q-major split dQ requires BF16 and head_dim=128")
    if block_n not in (32, 64):
        raise ValueError("block_n must be 32 or 64")

    batch, num_heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    num_q_blocks = (seqlen_q + 63) // 64
    if tuple(q2k_block_index.shape[:3]) != (batch, num_heads, num_q_blocks):
        raise ValueError("q2k_block_index has incompatible geometry")

    has_variable_counts = q2k_block_nums is not None
    counts = q2k_block_nums if has_variable_counts else q2k_block_index
    if has_variable_counts:
        counts = counts.contiguous()

    has_block_sizes = block_sizes is not None
    sizes = block_sizes if has_block_sizes else q2k_block_index
    if has_block_sizes:
        sizes = sizes.contiguous()
        if sizes.ndim == 1:
            stride_bsb, stride_bsk = 0, sizes.stride(0)
        elif sizes.ndim == 2 and tuple(sizes.shape) == (batch, (seqlen_k + 63) // 64):
            stride_bsb, stride_bsk = sizes.stride(0), sizes.stride(1)
        else:
            raise ValueError("block_sizes must have shape [K_blocks] or [B, K_blocks]")
    else:
        stride_bsb = stride_bsk = 0

    grid = (batch * num_heads * num_q_blocks,)
    _bsa_dq_qmajor_kernel[grid](
        q,
        k,
        v,
        dout,
        lse,
        neg_delta,
        q2k_block_index,
        counts,
        sizes,
        dq,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *dout.stride(),
        *lse.stride(),
        *neg_delta.stride(),
        *q2k_block_index.stride(),
        *(counts.stride() if has_variable_counts else (0, 0, 0)),
        stride_bsb,
        stride_bsk,
        *dq.stride(),
        seqlen_q,
        seqlen_k,
        num_heads,
        num_q_blocks,
        int(block_sparse_num),
        float(softmax_scale),
        HAS_VARIABLE_COUNTS=has_variable_counts,
        HAS_BLOCK_SIZES=has_block_sizes,
        BLOCK_M=64,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        num_warps=8,
        num_stages=2,
    )
    return dq


__all__ = ["bsa_dq_qmajor_triton", "qmajor_dq_available"]
