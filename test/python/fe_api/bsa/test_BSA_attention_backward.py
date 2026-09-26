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
def test_bsa_attention_backward_sm100_blk64():
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("blk64 backward test is specific to SM100/SM110")

    BSA = _import_bsa()
    block_size = 64
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

    # Outputs may alias metadata or inputs. The unique-writer fast path must
    # preserve those tensors until the backward kernel has finished reading.
    dk_storage = torch.empty_like(k)
    sizes_in_dk = dk_storage.view(torch.int32).flatten()[: block_sizes.numel()]
    sizes_in_dk.copy_(block_sizes)
    metadata_aliased = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        sizes_in_dk,
        dk_tensor=dk_storage,
        sparse_block_size=64,
    )
    torch.testing.assert_close(metadata_aliased["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)

    aliased = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        block_sizes,
        dk_tensor=k,
        sparse_block_size=64,
    )
    assert aliased["dk_tensor"].data_ptr() == k.data_ptr()
    torch.testing.assert_close(k.float(), dk_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
@pytest.mark.parametrize("bucket_size_blocks", [4, 1])
def test_bsa_attention_backward_sm100_blk64_variable_blocks_bucketed_accumulation(bucket_size_blocks):
    """Check dK/dV accumulation within one Q group and across Q groups."""
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("blk64 bucketed backward regression is specific to SM100/SM103")

    BSA = _import_bsa()
    block_size = 64
    batch, heads, dim = 1, 1, 128
    seqlen_q = 4 * block_size
    seqlen_k = 3 * block_size + 9
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)

    # KV0 is shared by three Q blocks. With bucket_size_blocks=1, three
    # different CTAs must add their dK/dV contributions to the same addresses.
    # The second Q block is empty, so its group has no edges at all. KV1
    # appears only in inactive suffixes; KV2 has a short logical block, and
    # KV3 is a short physical tail.
    q2k = torch.tensor([[[[0, 2, 1], [1, 0, 2], [2, 0, 3], [0, 3, 1]]]], device="cuda", dtype=torch.int32)
    q2k_block_nums = torch.tensor([[[2, 0, 3, 2]]], device="cuda", dtype=torch.int32)
    block_sizes = torch.tensor([64, 64, 37, 9], device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, 3, block_sizes, seqlen_q, seqlen_k, block_size, q2k_block_nums)
    # Dense softmax is NaN on an all-masked row. Remove the empty Q block
    # from the reference, then restore its mathematically zero dQ rows.
    active_q = torch.cat((q[:, :, :block_size], q[:, :, 2 * block_size :]), dim=2)
    active_do = torch.cat((do[:, :, :block_size], do[:, :, 2 * block_size :]), dim=2)
    active_mask = torch.cat((mask[:, :, :block_size], mask[:, :, 2 * block_size :]), dim=2)
    _, _, active_dq_ref, dk_ref, dv_ref = attention_backward_reference(active_q, k, v, active_do, active_mask)
    dq_ref = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dq_ref[:, :, :block_size] = active_dq_ref[:, :, :block_size]
    dq_ref[:, :, 2 * block_size :] = active_dq_ref[:, :, block_size:]

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=3,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_block_nums,
        sparse_block_size=block_size,
        allow_empty_block_nums=True,
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
        block_sparse_num=3,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_block_nums,
        sparse_block_size=block_size,
        bucket_size_blocks=bucket_size_blocks,
    )
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)

    torch.testing.assert_close(backward["dq_tensor"][:, :, block_size : 2 * block_size], torch.zeros_like(q[:, :, block_size : 2 * block_size]), atol=0, rtol=0)
    for name in ("dk_tensor", "dv_tensor"):
        grad = backward[name]
        torch.testing.assert_close(grad[:, :, 64:128], torch.zeros_like(grad[:, :, 64:128]), atol=0, rtol=0)
        invalid_kv2 = grad[:, :, 2 * block_size + 37 : 3 * block_size]
        torch.testing.assert_close(invalid_kv2, torch.zeros_like(invalid_kv2), atol=0, rtol=0)


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
@pytest.mark.parametrize("bucket_size_blocks", [8, 2])
def test_bsa_attention_backward_sm100_blk64_multiple_q_pairs_per_k_task(bucket_size_blocks):
    """Exercise repeated dQ pipeline phases and cross-CTA dK/dV accumulation."""
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("blk64 bucketed backward regression is specific to SM100/SM103")

    BSA = _import_bsa()
    block_size = 64
    q = torch.randn((1, 1, 8 * block_size, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, block_size, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    q2k = torch.zeros((1, 1, 8, 1), device="cuda", dtype=torch.int32)
    q2k_block_nums = torch.ones((1, 1, 8), device="cuda", dtype=torch.int32)
    block_sizes = torch.full((1,), block_size, device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, 1, block_sizes, q.shape[2], k.shape[2], block_size, q2k_block_nums)
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)

    forward = BSA.block_sparse_attention_forward(
        q, k, v, q2k, block_sparse_num=1, block_sizes=block_sizes,
        q2k_block_nums=q2k_block_nums, sparse_block_size=block_size, use_clc=False,
    )
    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k,
        block_sparse_num=1, block_sizes=block_sizes, q2k_block_nums=q2k_block_nums,
        sparse_block_size=block_size, bucket_size_blocks=bucket_size_blocks,
    )
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)


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
