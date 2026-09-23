"""Three attention workloads, with separate forward/backward CUDA timings."""

import argparse
from importlib.metadata import version
import statistics

import torch

from masks import (
    block_causal_metadata,
    build_bsa_metadata,
    build_geometry,
    count_visible_pairs,
    load_fixture,
    valid_token_mask,
)


BATCH_SIZE = 1
NUM_HEADS = 4
HEAD_DIM = 128


def measure(fn, label, warmup, runs, flush):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(runs):
        flush.add_(1)
        torch.cuda.synchronize()
        # Start/end NVTX ranges also cover autograd's worker thread.
        marker = torch.cuda.nvtx.range_start(label)
        start.record()
        result = fn()
        end.record()
        torch.cuda.nvtx.range_end(marker)
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result
    return statistics.median(samples)


def run_case(case, args, flush):
    seqlen = args.seqlen
    if case == "production":
        fixture = load_fixture()
        seqlen = fixture["padded_tokens"]
    torch.manual_seed(0)
    q, k, v = [
        torch.randn((BATCH_SIZE, seqlen, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]
    if case == "production":
        geometry = build_geometry(fixture["samples"], q.device)
        valid = valid_token_mask(geometry["cube_sizes"]).view(1, seqlen, 1, 1)
        for tensor in (q, k, v):
            tensor.masked_fill_(~valid, 0)
        metadata = build_bsa_metadata(q, k, geometry)
    elif case == "bsa-causal":
        metadata = block_causal_metadata(seqlen, NUM_HEADS, q.device)

    torch.manual_seed(1)
    do = torch.randn_like(q)
    if case == "causal":
        from flash_attn_cute import flash_attn_func

        q, k, v = [x.requires_grad_(True) for x in (q, k, v)]

        def forward():
            return flash_attn_func(
                q, k, v, causal=True, num_splits=1,
                pack_gqa=False, return_lse=True,
            )

        out, lse = forward()

        def backward():
            return torch.autograd.grad(out, (q, k, v), do, retain_graph=True)

    else:
        from cudnn import BSA

        def forward():
            return BSA.block_sparse_attention_forward(
                q, k, v, **metadata, sparse_block_size=64, layout="bshd",
            )

        out, lse = forward()
        if case == "production":
            # Padded query outputs are discarded by production's gather.
            do.masked_fill_(~valid, 0)

        use_block_causal_fastpath = case == "bsa-causal" and args.bsa_causal_bwd_backend == "flex"
        # ``blk64`` is the benchmark-facing name for the fused K-major BSA
        # implementation.  ``flex`` also enters the public API as ``fused``;
        # the explicit block_causal contract below selects its dedicated path.
        backward_backend = "split" if args.bsa_causal_bwd_backend == "split" else "fused"
        # The exact fast path derives the natural tail block from S.  Keep the
        # block_sizes tensor for forward/FLOP accounting and omit it only from
        # this backward call so arbitrary ragged metadata cannot be ignored.
        backward_metadata = (
            {name: value for name, value in metadata.items() if name != "block_sizes"}
            if use_block_causal_fastpath
            else metadata
        )

        def backward():
            return BSA.block_sparse_attention_backward(
                do, q, k, v, out, lse, **backward_metadata,
                sparse_block_size=64, layout="bshd",
                block_causal=use_block_causal_fastpath,
                backward_backend=backward_backend,
                qmajor_block_n=args.qmajor_block_n,
                bucket_size_blocks=args.bucket_size_blocks,
            )

        if (
            case == "bsa-causal"
            and args.bsa_causal_bwd_backend in ("flex", "split")
            and args.verify_fastpath
        ):
            candidate = backward()
            baseline = BSA.block_sparse_attention_backward(
                do,
                q,
                k,
                v,
                out,
                lse,
                **metadata,
                sparse_block_size=64,
                layout="bshd",
                block_causal=False,
                backward_backend="fused",
                bucket_size_blocks=args.bucket_size_blocks,
            )
            torch.cuda.synchronize()
            for name in ("dq_tensor", "dk_tensor", "dv_tensor"):
                candidate_grad = candidate[name]
                baseline_grad = baseline[name]
                max_abs = (candidate_grad.float() - baseline_grad.float()).abs().max().item()
                torch.testing.assert_close(
                    candidate_grad.float(),
                    baseline_grad.float(),
                    atol=5e-2,
                    rtol=5e-2,
                )
                print(
                    f"{args.bsa_causal_bwd_backend} backend check {name}: "
                    f"max_abs={max_abs:.6g}",
                    flush=True,
                )
            del candidate, baseline

        if case == "production" and args.verify_production_backend:
            candidate = backward()
            baseline = BSA.block_sparse_attention_backward(
                do,
                q,
                k,
                v,
                out,
                lse,
                **metadata,
                sparse_block_size=64,
                layout="bshd",
                backward_backend="fused",
                bucket_size_blocks=args.bucket_size_blocks,
            )
            torch.cuda.synchronize()
            for name in ("dq_tensor", "dk_tensor", "dv_tensor"):
                candidate_grad = candidate[name]
                baseline_grad = baseline[name]
                max_abs = (candidate_grad.float() - baseline_grad.float()).abs().max().item()
                torch.testing.assert_close(
                    candidate_grad.float(),
                    baseline_grad.float(),
                    atol=5e-2,
                    rtol=5e-2,
                )
                print(f"production backend check {name}: max_abs={max_abs:.6g}", flush=True)
            del candidate, baseline

    fwd_ms = measure(forward, f"BSA_ut__{case}__fwd", args.warmup, args.runs, flush)
    bwd_ms = measure(backward, f"BSA_ut__{case}__bwd", args.warmup, args.runs, flush)
    # Count after timing: metadata analysis must not enter the measured region.
    if case == "causal":
        pairs = BATCH_SIZE * NUM_HEADS * seqlen * (seqlen + 1) // 2
    else:
        pairs = count_visible_pairs(metadata)
    print_result(case, seqlen, pairs, fwd_ms, bwd_ms, args.peak_tflops)


def print_result(case, seqlen, pairs, fwd_ms, bwd_ms, peak_tflops):
    # Backward follows the five-matmul convention, including QK recompute.
    fwd_tflops = 4 * HEAD_DIM * pairs / (fwd_ms * 1e9)
    bwd_tflops = 10 * HEAD_DIM * pairs / (bwd_ms * 1e9)
    fwd_mfu = "—" if peak_tflops is None else f"{100 * fwd_tflops / peak_tflops:.2f}"
    bwd_mfu = "—" if peak_tflops is None else f"{100 * bwd_tflops / peak_tflops:.2f}"
    print(f"{case:12s}  {seqlen:7d}  {fwd_ms:10.3f}  {bwd_ms:10.3f}  "
          f"{fwd_tflops:12.3f}  {bwd_tflops:12.3f}  {fwd_mfu:>10s}  {bwd_mfu:>17s}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("causal", "bsa-causal", "production"), required=True)
    parser.add_argument("--seqlen", type=int, default=255424, help="Causal cases only; production uses production.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--bsa-causal-bwd-backend",
        choices=("blk64", "flex", "split"),
        default=None,
        help=(
            "Backward implementation for bsa-causal or production: the general "
            "64x64 fused BSA kernel, the exact block-causal Flex/FA4 fast path, "
            "or the exact Q-major split dQ path (defaults: bsa-causal=flex, "
            "production=split; production does not support flex)"
        ),
    )
    parser.add_argument(
        "--verify-bsa-causal-backend",
        "--verify-fastpath",
        dest="verify_fastpath",
        action="store_true",
        help="Compare the selected Flex/split bsa-causal backward with the blk64 baseline before timing",
    )
    parser.add_argument(
        "--qmajor-block-n",
        type=int,
        choices=(32, 64),
        default=32,
        help="K-token sub-tile for the split Q-major dQ kernel",
    )
    parser.add_argument(
        "--bucket-size-blocks",
        type=int,
        help="Override the blk64 dK/dV bucket size (try 512/1024/2048/3991)",
    )
    parser.add_argument(
        "--verify-production-backend",
        action="store_true",
        help="Compare the selected production backend with fused blk64 before timing",
    )
    peak = parser.add_mutually_exclusive_group()
    peak.add_argument("--peak-tflops", type=float, help="Explicit per-GPU dense BF16 peak TFLOP/s")
    peak.add_argument("--clock-mhz", type=float, help="GB200/SM100 locked clock; derive peak from runtime SM count (does not lock clocks)")
    args = parser.parse_args()
    if args.case == "causal":
        if args.bsa_causal_bwd_backend is not None:
            parser.error("--bsa-causal-bwd-backend applies only to --case bsa-causal or production")
    elif args.bsa_causal_bwd_backend is None:
        args.bsa_causal_bwd_backend = "flex" if args.case == "bsa-causal" else "split"
    if args.case == "production" and args.bsa_causal_bwd_backend == "flex":
        parser.error(
            "the Flex backend supports only the exact block-causal-64 mask; "
            "use blk64 or split for --case production"
        )
    if args.verify_fastpath:
        if args.case != "bsa-causal":
            parser.error("--verify-bsa-causal-backend requires --case bsa-causal")
        if args.bsa_causal_bwd_backend == "blk64":
            parser.error("blk64 is already the bsa-causal verification baseline")
    if args.verify_production_backend and args.case != "production":
        parser.error("--verify-production-backend requires --case production")
    if args.verify_production_backend and args.bsa_causal_bwd_backend == "blk64":
        parser.error("blk64 is already the production verification baseline")
    if (
        args.case == "bsa-causal"
        and args.bsa_causal_bwd_backend == "flex"
        and args.bucket_size_blocks is not None
    ):
        parser.error("--bucket-size-blocks does not apply to the Flex block-causal fast path")
    if args.seqlen <= 0 or args.warmup < 1 or args.runs < 1:
        parser.error("seqlen, warmup and runs must be positive")
    if args.peak_tflops is not None and not (0 < args.peak_tflops < float("inf")):
        parser.error("peak-tflops must be positive and finite")
    if args.clock_mhz is not None:
        if not (0 < args.clock_mhz < float("inf")):
            parser.error("clock-mhz must be positive and finite")
        if torch.cuda.get_device_capability() != (10, 0):
            parser.error("clock-mhz uses the GB200/SM100 peak model; use peak-tflops for other GPUs")
        sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        # SM100 dense BF16 model: 8192 FLOPs / SM / cycle (FMA counts as 2).
        args.peak_tflops = sm_count * 8192 * args.clock_mhz / 1e6
        print(f"GB200/SM100 peak model: {sm_count} SM x 8192 FLOPs/cycle x {args.clock_mhz:g} MHz "
              f"= {args.peak_tflops:.3f} TFLOP/s")
    print(f"GPU: {torch.cuda.get_device_name()}; Torch: {torch.__version__}; CUDA: {torch.version.cuda}")
    package = "flash-attn-cute" if args.case == "causal" else "nvidia-cudnn-frontend"
    print(f"{package}: {version(package)}; CUTLASS DSL: {version('nvidia-cutlass-dsl')}")
    print(f"BF16 BSHD, B={BATCH_SIZE} H={NUM_HEADS} D={HEAD_DIM}; median milliseconds; L2 flushed per sample")
    if args.case == "bsa-causal":
        suffix = (
            f"; qmajor_block_n={args.qmajor_block_n}; "
            f"bucket_size_blocks={args.bucket_size_blocks}"
            if args.bsa_causal_bwd_backend == "split"
            else ""
        )
        print(f"BSA causal backward backend: {args.bsa_causal_bwd_backend}{suffix}")
    if args.case == "production":
        print(
            f"BSA production backward backend: {args.bsa_causal_bwd_backend}; "
            f"qmajor_block_n={args.qmajor_block_n}; bucket_size_blocks={args.bucket_size_blocks}"
        )
    if args.peak_tflops is not None:
        print(f"Single-GPU dense BF16 peak: {args.peak_tflops:.3f} TFLOP/s")
    print("Bwd TFLOP/s and MFU include QK recompute (10*D*pairs); fwd uses 4*D*pairs")
    print(f"{'case':12s}  {'seqlen':>7s}  {'fwd_ms':>10s}  {'bwd_ms':>10s}  "
          f"{'fwd_TFLOP/s':>12s}  {'bwd_TFLOP/s':>12s}  {'fwd_MFU_%':>10s}  {'bwd_MFU_recomp_%':>17s}")
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.zeros(max(2 * l2_bytes, 256 * 1024**2) // 4, dtype=torch.int32, device="cuda")
    run_case(args.case, args, flush)


if __name__ == "__main__":
    main()
