#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT=${BSA_ROOT:-/home/scratch.xshang_wwfo/BSA_Kernel}
NCU=${NCU:-$ROOT/profiler/nsight_compute-linux-x86_64-2026.3.0.13-archive/ncu}
PYTHON=${BSA_PYTHON:-$ROOT/.venv/bsa/bin/python}
PROFILE_PREFIX=${PROFILE_PREFIX:-$ROOT/profiler/bsa_bwd_$(date +%Y%m%d_%H%M%S)_$$}
export XDG_CACHE_HOME=/tmp/BSA_xshang_cache
export TRITON_CACHE_DIR=/tmp/BSA_xshang_cache/triton

mkdir -p "$(dirname "$PROFILE_PREFIX")"
cd "$ROOT/cudnn-frontend/BSA_ut/BSA_ut"
echo "Nsight Compute output prefix: $PROFILE_PREFIX"

if [[ "${1:-}" == "permission" ]]; then
    "$NCU" --set basic --launch-count 1 "$PYTHON" -c 'import torch; torch.empty(1024, device="cuda").fill_(1)'
else
    "$NCU" \
        --target-processes all \
        --set basic \
        --section WarpStateStats \
        --section SourceCounters \
        --kernel-name 'regex:^kernel_cutlass_bwd_.*' \
        --launch-count 1 \
        --page source \
        --print-source sass \
        --csv \
        --log-file "${PROFILE_PREFIX}_source.csv" \
        --export "${PROFILE_PREFIX}.ncu-rep" \
        -f \
        "$PYTHON" -u benchmark.py \
        --case production \
        --warmup 1 \
        --runs 1 \
        --bsa-causal-bwd-backend blk64
fi
