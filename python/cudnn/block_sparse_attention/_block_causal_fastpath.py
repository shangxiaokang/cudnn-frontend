# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact 64-token block-causal backward fast path.

The general BSA backward kernel is K-major and optimized for arbitrary sparse
graphs.  A regular block-causal graph has much more structure: query token q
can see every key in its 64-token block and in all preceding blocks.  Flex
Attention's interval plan represents that predicate exactly and lets its
SM100 128x128 backward kernel classify almost every tile as fully visible;
only the 64x64 upper-right quadrant of each diagonal 128x128 tile is partial.

This module is intentionally private.  The public BSA API exposes the route
through an explicit ``block_causal=True`` contract so arbitrary metadata can
never be silently reinterpreted as block causal.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from cudnn.api_base import TupleDict


_BLOCK_SIZE = 64
_PLAN_CACHE_CAPACITY = 4
_PLAN_CACHE: OrderedDict[tuple, object] = OrderedDict()


def _as_bshd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor if layout == "bshd" else tensor.transpose(1, 2)


def _plan_key(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple:
    major, minor = torch.cuda.get_device_capability(q.device)
    return (
        q.device.type,
        q.device.index,
        major * 10 + minor,
        q.dtype,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
    )


def _make_block_causal_endpoints(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Return [1, 1, B*Sq] endpoints for floor(k/64) <= floor(q/64)."""

    q_position = torch.arange(seqlen_q, dtype=torch.int32, device=device)
    row_end = ((q_position // _BLOCK_SIZE) + 1) * _BLOCK_SIZE
    row_end.clamp_(max=seqlen_k)
    return row_end.repeat(batch_size).view(1, 1, batch_size * seqlen_q)


def _get_block_causal_plan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    key = _plan_key(q, k, v)
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        _PLAN_CACHE.move_to_end(key)
        return plan

    # Import lazily: ordinary BSA users should not pay Flex Attention's import
    # or initialization cost, and the dependency remains one-way at runtime.
    from cudnn.flex_attention.plan.builder import create_mask_plan

    batch_size, seqlen_q, _, _ = q.shape
    seqlen_k = k.shape[1]
    endpoints = _make_block_causal_endpoints(batch_size, seqlen_q, seqlen_k, q.device)
    plan = create_mask_plan(
        endpoints,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    _PLAN_CACHE[key] = plan
    _PLAN_CACHE.move_to_end(key)
    while len(_PLAN_CACHE) > _PLAN_CACHE_CAPACITY:
        _PLAN_CACHE.popitem(last=False)
    return plan


def block_causal_64_backward(
    do_tensor: torch.Tensor,
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    *,
    softmax_scale: float | None,
    dq_tensor: torch.Tensor | None,
    dk_tensor: torch.Tensor | None,
    dv_tensor: torch.Tensor | None,
    layout: str,
) -> TupleDict:
    """Execute exact block-causal-64 backward through the packed-mask kernel."""

    from cudnn.flex_attention.execution import _flex_attention_backward

    q_bshd, k_bshd, v_bshd, o_bshd, do_bshd = (
        _as_bshd(tensor, layout)
        for tensor in (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor)
    )
    plan = _get_block_causal_plan(q_bshd, k_bshd, v_bshd)
    with torch.cuda.nvtx.range("bsa_block_causal_64_flex_bwd"):
        result = _flex_attention_backward(
            q_bshd,
            k_bshd,
            v_bshd,
            o_bshd,
            do_bshd,
            lse_tensor,
            mask_plan=plan,
            softmax_scale=softmax_scale,
            deterministic=False,
        )

    dq_fast, dk_fast, dv_fast = result
    if layout == "bhsd":
        dq_fast, dk_fast, dv_fast = (
            tensor.transpose(1, 2) for tensor in (dq_fast, dk_fast, dv_fast)
        )

    outputs = []
    for fast, requested in (
        (dq_fast, dq_tensor),
        (dk_fast, dk_tensor),
        (dv_fast, dv_tensor),
    ):
        if requested is None:
            outputs.append(fast)
        else:
            requested.detach().copy_(fast)
            outputs.append(requested)
    return TupleDict(dq_tensor=outputs[0], dk_tensor=outputs[1], dv_tensor=outputs[2])


__all__ = ["block_causal_64_backward"]
