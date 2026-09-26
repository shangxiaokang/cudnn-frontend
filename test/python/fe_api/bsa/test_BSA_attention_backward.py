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

    if num_q_blocks == 2:
        # Metadata can also be a view into a caller-provided output buffer.
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

        # The caller may reuse K storage for dK. The direct-output path must
        # preserve K until the last backward kernel finishes reading it.
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
@torch_fork_set_rng(seed=17)
@pytest.mark.parametrize("qmajor_block_n", [32, 64])
@pytest.mark.parametrize("bucket_size_blocks", [None, 2])
def test_bsa_attention_backward_sm100_blk64_split_irregular(
    qmajor_block_n,
    bucket_size_blocks,
):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("split blk64 backward is specific to SM100/SM103")
    pytest.importorskip("triton")

    BSA = _import_bsa()
    block_size = 64
    batch, heads, q_blocks, kv_blocks, dim = 2, 2, 6, 7, 128
    seqlen_q, seqlen_k = q_blocks * block_size, kv_blocks * block_size
    q = torch.randn(
        (batch, seqlen_q, heads, dim), device="cuda", dtype=torch.bfloat16
    )
    k, v = [
        torch.randn(
            (batch, seqlen_k, heads, dim), device="cuda", dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    do = torch.randn_like(q)

    # Head-specific rows and a capacity larger than several active prefixes
    # exercise runtime counts, odd tails, and arbitrary (not causal) ordering.
    q2k = torch.zeros(
        (batch, heads, q_blocks, kv_blocks), dtype=torch.int32, device="cuda"
    )
    counts = torch.tensor(
        [[[1, 2, 3, 4, 5, 3], [5, 4, 3, 2, 1, 2]]],
        dtype=torch.int32,
        device="cuda",
    ).expand(batch, -1, -1).contiguous()
    # Keep the final K block completely unreferenced.  This checks that both
    # the unique-writer store and multi-group atomic paths preserve the zero
    # gradient produced by the pre-zeroed dK/dV accumulator workspace.
    ids = torch.arange(kv_blocks - 1, dtype=torch.int32, device="cuda")
    counts_host = ([1, 2, 3, 4, 5, 3], [5, 4, 3, 2, 1, 2])
    for batch_idx in range(batch):
        for row in range(q_blocks):
            count_h0 = counts_host[0][row]
            count_h1 = counts_host[1][row]
            q2k[batch_idx, 0, row, :count_h0] = ids.roll(batch_idx).flip(0)[:count_h0]
            q2k[batch_idx, 1, row, :count_h1] = ids.roll(row + batch_idx)[:count_h1]
    # K block sizes are shared across batches by the public API. Preserve
    # irregular sizes while checking distinct sparse rows for each batch.
    block_sizes = torch.tensor(
        [64, 30, 48, 40, 13, 51, 29],
        dtype=torch.int32,
        device="cuda",
    )

    mask = block_sparse_mask(
        q2k,
        kv_blocks,
        block_sizes,
        seqlen_q,
        seqlen_k,
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
        kv_blocks,
        block_sizes,
        q2k_block_nums=counts,
        sparse_block_size=block_size,
        layout="bshd",
        use_clc=False,
    )
    dq, dk, dv = (
        torch.full_like(q, float("nan")),
        torch.full_like(k, float("nan")),
        torch.full_like(v, float("nan")),
    )
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        kv_blocks,
        block_sizes,
        q2k_block_nums=counts,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        sparse_block_size=block_size,
        layout="bshd",
        backward_backend="split",
        qmajor_block_n=qmajor_block_n,
        bucket_size_blocks=bucket_size_blocks,
    )
    assert backward["dq_tensor"].data_ptr() == dq.data_ptr()
    assert backward["dk_tensor"].data_ptr() == dk.data_ptr()
    assert backward["dv_tensor"].data_ptr() == dv.data_ptr()
    torch.testing.assert_close(dq.transpose(1, 2).float(), dq_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dk.transpose(1, 2).float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dv.transpose(1, 2).float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
def test_bucketed_k2q_csr_long_rows_and_dynamic_capacity():
    if not torch.cuda.is_available():
        pytest.skip("bucketed K2Q CSR test requires CUDA")
    if torch.cuda.get_device_capability()[0] not in {9, 10, 11}:
        pytest.skip("bucketed K2Q CSR CuTe kernel requires SM90-SM110")

    try:
        from cudnn.block_sparse_attention.csrc.bwd.bucketed_k2q_csr import (
            build_bucketed_k2q_csr_cutedsl,
        )
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")

    batch, heads, q_blocks, kv_blocks = 1, 2, 4, 320

    def check_capacity(capacity, counts_values):
        base = torch.arange(capacity, dtype=torch.int32, device="cuda")
        q2k = torch.empty(
            (batch, heads, q_blocks, capacity),
            dtype=torch.int32,
            device="cuda",
        )
        for head in range(heads):
            for row in range(q_blocks):
                q2k[0, head, row] = base.roll(17 * row + 31 * head)
        counts = torch.tensor(
            [counts_values],
            dtype=torch.int32,
            device="cuda",
        )

        offsets, indices, num_groups, _ = build_bucketed_k2q_csr_cutedsl(
            q2k,
            capacity,
            kv_blocks,
            bucket_size_blocks=2,
            q2k_block_nums=counts,
        )
        torch.cuda.synchronize()
        assert num_groups == 2

        q2k_cpu = q2k.cpu()
        counts_cpu = counts.cpu()
        offsets_cpu = offsets.cpu()
        indices_cpu = indices.cpu()
        active_ids = {}
        for head in range(heads):
            for q_block in range(q_blocks):
                count = int(counts_cpu[0, head, q_block])
                active_ids[head, q_block] = set(
                    q2k_cpu[0, head, q_block, :count].tolist()
                )
        for head in range(heads):
            for group in range(num_groups):
                q_begin, q_end = group * 2, min((group + 1) * 2, q_blocks)
                for kv_block in range(kv_blocks):
                    begin = int(offsets_cpu[0, head, group, kv_block])
                    end = int(offsets_cpu[0, head, group, kv_block + 1])
                    actual = sorted(indices_cpu[0, head, begin:end].tolist())
                    expected = [
                        q_block
                        for q_block in range(q_begin, q_end)
                        if kv_block in active_ids[head, q_block]
                    ]
                    assert actual == expected

    # The second call must reuse the variable-count compile-cache entry even
    # though the physical row capacity changes, and it must execute the
    # stride-256 loop at least twice for several rows.
    check_capacity(17, ((17, 13, 5, 0), (13, 17, 0, 5)))
    check_capacity(300, ((300, 257, 17, 0), (257, 300, 0, 17)))

    # Fixed-count rows have no inactive capacity to skip and retain the old
    # capacity-tiled grid.  Exercise its second 256-edge CTA explicitly.
    capacity = 300
    base = torch.arange(capacity, dtype=torch.int32, device="cuda")
    q2k_fixed = torch.stack((base, base.roll(37))).view(1, 1, 2, capacity)
    offsets, indices, num_groups, _ = build_bucketed_k2q_csr_cutedsl(
        q2k_fixed,
        capacity,
        kv_blocks,
        bucket_size_blocks=1,
    )
    torch.cuda.synchronize()
    assert num_groups == 2
    offsets_cpu = offsets.cpu()
    indices_cpu = indices.cpu()
    q2k_fixed_cpu = q2k_fixed.cpu()
    for group in range(num_groups):
        active_ids = set(q2k_fixed_cpu[0, 0, group].tolist())
        for kv_block in range(kv_blocks):
            begin = int(offsets_cpu[0, 0, group, kv_block])
            end = int(offsets_cpu[0, 0, group, kv_block + 1])
            actual = indices_cpu[0, 0, begin:end].tolist()
            expected = [group] if kv_block in active_ids else []
            assert actual == expected


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
