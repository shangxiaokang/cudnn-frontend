# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.bsa.bsa_reference import attention_backward_reference, block_sparse_mask
from fe_api.bsa.bsa_utils import make_fixed_metadata, supported_block_size

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _import_bsa():
    try:
        from cudnn import BSA

        importlib.import_module("cudnn.block_sparse_attention._interface")

        return BSA
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
def test_bsa_attention_backward_fixed_blocks():
    BSA = _import_bsa()
    block_size = supported_block_size(backward=True)
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=block_size,
    )
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        None,
        sparse_block_size=block_size,
    )
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)

    q_bshd, k_bshd, v_bshd = (tensor.transpose(1, 2) for tensor in (q, k, v))
    do_bshd = do.transpose(1, 2)
    forward_bshd = BSA.block_sparse_attention_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=block_size,
        layout="bshd",
    )
    dq_bshd = torch.empty_like(q_bshd)
    dk_bshd = torch.empty_like(k_bshd)
    dv_bshd = torch.empty_like(v_bshd)
    backward_bshd = BSA.block_sparse_attention_backward(
        do_bshd,
        q_bshd,
        k_bshd,
        v_bshd,
        forward_bshd["o_tensor"],
        forward_bshd["lse_tensor"],
        q2k,
        block_sparse_num,
        None,
        dq_tensor=dq_bshd,
        dk_tensor=dk_bshd,
        dv_tensor=dv_bshd,
        sparse_block_size=block_size,
        layout="bshd",
    )
    assert backward_bshd["dq_tensor"].data_ptr() == dq_bshd.data_ptr()
    assert backward_bshd["dk_tensor"].data_ptr() == dk_bshd.data_ptr()
    assert backward_bshd["dv_tensor"].data_ptr() == dv_bshd.data_ptr()
    torch.testing.assert_close(backward_bshd["dq_tensor"].transpose(1, 2).float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward_bshd["dk_tensor"].transpose(1, 2).float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward_bshd["dv_tensor"].transpose(1, 2).float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
@pytest.mark.parametrize("num_q_blocks", [2, 3, 6])
def test_bsa_attention_backward_sm100_blk64(num_q_blocks):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("blk64 backward test is specific to SM100/SM110")

    BSA = _import_bsa()
    block_size = 64
    # The selected K blocks receive num_q_blocks edges.  Two covers the
    # prologue/epilogue-only path; three and six exercise the pipelined main
    # loop with odd and even edge counts, including the dQ-to-next-dP handoff.
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, num_q_blocks * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=64,
        use_clc=False,
    )
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=64,
    )
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=17)
@pytest.mark.parametrize("qmajor_block_n", [32, 64])
def test_bsa_attention_backward_sm100_blk64_split_irregular(qmajor_block_n):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("split blk64 backward is specific to SM100/SM103")
    pytest.importorskip("triton")

    BSA = _import_bsa()
    block_size = 64
    batch, heads, blocks, dim = 2, 2, 5, 128
    seqlen = blocks * block_size
    q, k, v = [
        torch.randn((batch, seqlen, heads, dim), device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    do = torch.randn_like(q)

    # Head-specific rows and a capacity larger than several active prefixes
    # exercise runtime counts, odd tails, and arbitrary (not causal) ordering.
    q2k = torch.zeros((batch, heads, blocks, blocks), dtype=torch.int32, device="cuda")
    counts = torch.tensor(
        [[[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]],
        dtype=torch.int32,
        device="cuda",
    ).expand(batch, -1, -1).contiguous()
    ids = torch.arange(blocks, dtype=torch.int32, device="cuda")
    for batch_idx in range(batch):
        for row in range(blocks):
            q2k[batch_idx, 0, row, : row + 1] = ids.roll(batch_idx).flip(0)[: row + 1]
            q2k[batch_idx, 1, row, : blocks - row] = ids.roll(row + batch_idx)[: blocks - row]
    block_sizes = torch.tensor(
        [[64, 30, 48, 40, 13], [55, 64, 23, 48, 32]],
        dtype=torch.int32,
        device="cuda",
    )

    mask = block_sparse_mask(
        q2k,
        blocks,
        block_sizes,
        seqlen,
        seqlen,
        block_size,
        q2k_block_nums=counts,
    )
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        do.transpose(1, 2),
        mask,
    )

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        blocks,
        block_sizes,
        q2k_block_nums=counts,
        sparse_block_size=block_size,
        layout="bshd",
        use_clc=False,
    )
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        blocks,
        block_sizes,
        q2k_block_nums=counts,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        sparse_block_size=block_size,
        layout="bshd",
        backward_backend="split",
        qmajor_block_n=qmajor_block_n,
    )
    assert backward["dq_tensor"].data_ptr() == dq.data_ptr()
    assert backward["dk_tensor"].data_ptr() == dk.data_ptr()
    assert backward["dv_tensor"].data_ptr() == dv.data_ptr()
    torch.testing.assert_close(dq.transpose(1, 2).float(), dq_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dk.transpose(1, 2).float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dv.transpose(1, 2).float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
@pytest.mark.parametrize("seqlen", [160, 192])
def test_bsa_attention_backward_block_causal_64_fastpath(seqlen):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("block-causal-64 fast path is specific to SM100/SM103")

    BSA = _import_bsa()
    block_size = 64
    # Three blocks exercise both the masked upper-right quadrant of a complete
    # 128x128 diagonal tile and the residual half of the final tile.  S=160
    # additionally checks endpoint clamping and a partially valid physical
    # block.
    batch, heads, dim = 1, 2, 128
    num_blocks = (seqlen + block_size - 1) // block_size
    q, k, v = [
        torch.randn((batch, seqlen, heads, dim), device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    do = torch.randn_like(q)
    block_ids = torch.arange(num_blocks, dtype=torch.int32, device="cuda")
    q2k = (
        block_ids.view(1, 1, 1, num_blocks)
        .expand(batch, heads, num_blocks, num_blocks)
        .contiguous()
    )
    q2k_block_nums = (
        (block_ids + 1)
        .view(1, 1, num_blocks)
        .expand(batch, heads, num_blocks)
        .contiguous()
    )
    block_sizes = torch.full(
        (num_blocks,), block_size, dtype=torch.int32, device="cuda"
    )
    block_sizes[-1] = seqlen - (num_blocks - 1) * block_size
    mask = block_sparse_mask(
        q2k,
        num_blocks,
        block_sizes,
        seqlen,
        seqlen,
        block_size,
        q2k_block_nums=q2k_block_nums,
    )
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        do.transpose(1, 2),
        mask,
    )

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        num_blocks,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        sparse_block_size=block_size,
        layout="bshd",
    )
    with pytest.raises(ValueError, match="requires block_sizes=None"):
        BSA.block_sparse_attention_backward(
            do,
            q,
            k,
            v,
            forward["o_tensor"],
            forward["lse_tensor"],
            q2k,
            num_blocks,
            block_sizes,
            q2k_block_nums=q2k_block_nums,
            sparse_block_size=block_size,
            layout="bshd",
            block_causal=True,
        )
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        num_blocks,
        None,
        q2k_block_nums=q2k_block_nums,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        sparse_block_size=block_size,
        layout="bshd",
        block_causal=True,
    )
    assert backward["dq_tensor"].data_ptr() == dq.data_ptr()
    assert backward["dk_tensor"].data_ptr() == dk.data_ptr()
    assert backward["dv_tensor"].data_ptr() == dv.data_ptr()
    torch.testing.assert_close(dq.transpose(1, 2).float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dk.transpose(1, 2).float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dv.transpose(1, 2).float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
@pytest.mark.parametrize("num_q_blocks", [1, 4])
def test_bsa_attention_backward_blk128_dk_zero_init_accumulate_transition(num_q_blocks):
    """dK zero-init on the first Q block and accumulate on later Q blocks.

    num_q_blocks=1 exercises pure zero-initialization; num_q_blocks=4 makes
    several Q blocks write the same K blocks, exercising the runtime
    initialize-to-accumulate transition of the dK MMA predicate.
    """
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("blk128 backward is specific to SM100/SM110")

    BSA = _import_bsa()
    block_size = 128
    batch, heads, dim = 1, 1, 128
    seqlen_q = num_q_blocks * block_size
    seqlen_k = 4 * block_size
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)

    # Every Q block attends to the same two K blocks.
    num_kv_blocks = seqlen_k // block_size
    selected = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    q2k = selected.view(1, 1, 1, 2).expand(batch, heads, num_q_blocks, 2).contiguous()
    block_sparse_num = 2
    full_block_sizes = torch.full((num_kv_blocks,), block_size, dtype=torch.int32, device="cuda")
    mask = block_sparse_mask(q2k, block_sparse_num, full_block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        None,
        sparse_block_size=block_size,
    )
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        None,
        sparse_block_size=block_size,
    )
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)
