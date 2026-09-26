# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep production BSA backward configurations on one fixed input and mask."""

import argparse
from functools import partial

import torch

import cudnn
from cudnn import BSA

from benchmark import BATCH_SIZE, HEAD_DIM, NUM_HEADS, measure
from masks import build_bsa_metadata, build_geometry, load_fixture, valid_token_mask


def make_workload():
    fixture = load_fixture()
    seqlen = int(fixture["padded_tokens"])
    torch.manual_seed(0)
    q, k, v = [
        torch.randn(
            (BATCH_SIZE, seqlen, NUM_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(3)
    ]
    geometry = build_geometry(fixture["samples"], q.device)
    valid = valid_token_mask(geometry["cube_sizes"]).view(1, seqlen, 1, 1)
    for tensor in (q, k, v):
        tensor.masked_fill_(~valid, 0)
    metadata = build_bsa_metadata(q, k, geometry)

    torch.manual_seed(1)
    do = torch.randn_like(q)
    do.masked_fill_(~valid, 0)
    out, lse = BSA.block_sparse_attention_forward(
        q, k, v, **metadata, sparse_block_size=64, layout="bshd"
    )
    return do, q, k, v, out, lse, metadata, seqlen


def check_gradients(candidate, baseline, label):
    for name in ("dq_tensor", "dk_tensor", "dv_tensor"):
        candidate_grad = candidate[name].float()
        baseline_grad = baseline[name].float()
        max_abs = (candidate_grad - baseline_grad).abs().max().item()
        if max_abs > 0.01:
            raise AssertionError(f"{label} {name}: max_abs={max_abs:.6g} exceeds 0.01")
        torch.testing.assert_close(
            candidate_grad, baseline_grad, atol=5e-2, rtol=5e-2
        )
        print(f"verify {label} {name}: max_abs={max_abs:.6g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--verify", action="store_true", help="Compare all gradients with fused bucket=full")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("fused", "split"),
        default=("fused", "split"),
        help="Backward backends to time; default: fused split",
    )
    parser.add_argument(
        "--buckets",
        "--split-buckets",
        dest="buckets",
        type=int,
        nargs="+",
        help="Fused and split Q-block buckets; default: full Q domain, 2048, 1024, 512, 256",
    )
    parser.add_argument(
        "--block-n",
        type=int,
        nargs="+",
        choices=(32, 64),
        default=(32, 64),
        help="Q-major dQ K-token sub-tile sizes",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.runs < 1:
        parser.error("warmup and runs must be positive")
    backends = set(args.backends)

    workload = make_workload()
    do, q, k, v, out, lse, metadata, seqlen = workload
    full_bucket = (seqlen + 63) // 64
    buckets = tuple(dict.fromkeys((full_bucket, *(args.buckets or (2048, 1024, 512, 256)))))
    if any(bucket <= 0 for bucket in buckets):
        parser.error("bucket sizes must be positive")

    def backward(backend, bucket, block_n=32):
        return BSA.block_sparse_attention_backward(
            do, q, k, v, out, lse, **metadata,
            sparse_block_size=64,
            layout="bshd",
            block_causal=False,
            backward_backend=backend,
            bucket_size_blocks=bucket,
            qmajor_block_n=block_n,
        )

    fused = partial(backward, "fused", full_bucket)
    configs = [
        ("fused", bucket, None, partial(backward, "fused", bucket))
        for bucket in buckets if bucket != full_bucket
    ] if "fused" in backends else []
    configs += [
        ("split", bucket, block_n, partial(backward, "split", bucket, block_n))
        for bucket in buckets for block_n in args.block_n
    ] if "split" in backends else []

    print(
        f"GPU: {torch.cuda.get_device_name()}; Torch: {torch.__version__}; "
        f"CUDA: {torch.version.cuda}; cuDNN Frontend: {cudnn.__file__}",
        flush=True,
    )
    print(
        f"production BF16 BSHD [{BATCH_SIZE}, {seqlen}, {NUM_HEADS}, {HEAD_DIM}]; "
        f"one QKV/mask/O/LSE; backends={','.join(sorted(backends))}; "
        f"warmup={args.warmup}; runs={args.runs}; "
        "CUDA Event median; L2 flushed before each sample",
        flush=True,
    )

    if args.verify:
        baseline = fused()
        for backend, bucket, block_n, fn in configs:
            label = f"{backend} bucket={bucket} block_n={block_n or '-'}"
            candidate = fn()
            check_gradients(candidate, baseline, label)
            del candidate
        del baseline
        torch.cuda.synchronize()

    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.zeros(
        max(2 * l2_bytes, 256 * 1024**2) // 4,
        dtype=torch.int32,
        device="cuda",
    )
    print(f"{'backend':>8s}  {'bucket':>6s}  {'block_n':>7s}  {'bwd_ms':>10s}  {'speedup':>8s}", flush=True)
    fused_ms = None
    if "fused" in backends:
        fused_ms = measure(
            fused, "BSA_ut__production__bwd__fused", args.warmup, args.runs, flush
        )
        print(f"{'fused':>8s}  {full_bucket:6d}  {'-':>7s}  {fused_ms:10.3f}  {'1.000x':>8s}", flush=True)
    for backend, bucket, block_n, fn in configs:
        label = f"BSA_ut__production__bwd__{backend}__bucket{bucket}__n{block_n or 0}"
        ms = measure(fn, label, args.warmup, args.runs, flush)
        speedup = "-" if fused_ms is None else f"{fused_ms / ms:.3f}x"
        print(
            f"{backend:>8s}  {bucket:6d}  {str(block_n or '-'):>7s}  {ms:10.3f}  "
            f"{speedup:>8s}",
            flush=True,
        )


if __name__ == "__main__":
    main()
