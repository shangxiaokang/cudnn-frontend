#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FLASH_ATTN_DIR="$SCRIPT_DIR/flash-attention"
FLASH_ATTN_URL="https://github.com/jiayus-nvidia/flash-attention.git"
FLASH_ATTN_BRANCH="magi_backend"
FLASH_ATTN_COMMIT="55221a93a8fc415a721502ed68643983dbf67862"

if [[ -e "$FLASH_ATTN_DIR" && ! -d "$FLASH_ATTN_DIR/.git" ]]; then
    echo "$FLASH_ATTN_DIR exists but is not a Git repository" >&2
    exit 1
fi

if [[ ! -d "$FLASH_ATTN_DIR/.git" ]]; then
    git clone \
        --branch "$FLASH_ATTN_BRANCH" \
        --single-branch \
        "$FLASH_ATTN_URL" \
        "$FLASH_ATTN_DIR"
    git -C "$FLASH_ATTN_DIR" checkout --detach "$FLASH_ATTN_COMMIT"
fi

echo "Using flash-attention commit: $(git -C "$FLASH_ATTN_DIR" rev-parse HEAD)"

python -m pip install -r requirements-causal.txt
# Fail before compiling extensions if the installed DSL cannot be imported.
python - <<'CHECK_DSL'
import sys

print("Python:", sys.executable, flush=True)
import cutlass.cute as cute
print("CUTLASS:", cute.__file__)
CHECK_DSL
python -m pip install \
    --no-build-isolation \
    --no-deps \
    "$FLASH_ATTN_DIR/csrc/utils/create_block_mask"
python -m pip install \
    --no-build-isolation \
    --no-deps \
    "$FLASH_ATTN_DIR/flash_attn/cute"

python - <<'PY'
from importlib.metadata import version

import cutlass.cute as cute
import create_block_mask_cuda
import flash_attn_cute
from flash_attn_cute import flash_attn_func

cute.make_fragment
print("nvidia-cutlass-dsl:", version("nvidia-cutlass-dsl"))
print("flash-attn-cute:", version("flash-attn-cute"))
print("flash_attn_cute:", flash_attn_cute.__file__)
print("create_block_mask_cuda:", create_block_mask_cuda.__file__)
print("Installation checks passed.")
PY
