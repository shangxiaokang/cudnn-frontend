#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT=${BSA_ROOT:-/home/scratch.xshang_wwfo/BSA_Kernel}
PYTHON=${BSA_PYTHON:-$ROOT/.venv/bsa/bin/python}
export XDG_CACHE_HOME=/tmp/BSA_xshang_cache
export TRITON_CACHE_DIR=/tmp/BSA_xshang_cache/triton
cd "$ROOT/cudnn-frontend/BSA_ut/BSA_ut"
exec "$PYTHON" -u sweep_production.py "$@"
