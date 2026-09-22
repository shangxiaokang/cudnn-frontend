#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDNN_DIR="$SCRIPT_DIR/cudnn-frontend"
CUDNN_URL="https://github.com/NVIDIA/cudnn-frontend.git"
CUDNN_TAG="v1.29.0"
CUDNN_COMMIT="91dbf3e976a161c1a6833198c708a449330dc3ce"

if [[ -e "$CUDNN_DIR" && ! -d "$CUDNN_DIR/.git" ]]; then
    echo "$CUDNN_DIR exists but is not a Git repository" >&2
    exit 1
fi

# Preserve an existing checkout so local kernel edits survive rebuilding.
if [[ ! -d "$CUDNN_DIR/.git" ]]; then
    git clone --branch "$CUDNN_TAG" --single-branch "$CUDNN_URL" "$CUDNN_DIR"
    git -C "$CUDNN_DIR" checkout --detach "$CUDNN_COMMIT"
fi
echo "Using cuDNN Frontend commit: $(git -C "$CUDNN_DIR" rev-parse HEAD)"

python -m pip install -r requirements.txt
# Fail before compiling cuDNN if the installed DSL cannot be imported.
python - <<'CHECK_DSL'
import sys

print("Python:", sys.executable, flush=True)
import cutlass.cute as cute
print("CUTLASS:", cute.__file__)
CHECK_DSL
export CUDNN_PATH
CUDNN_PATH="$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-cudnn-cu13").locate_file("nvidia/cudnn"))')"
unset CUDNN_INCLUDE_PATH CUDNN_LIBRARY_PATH
export CMAKE_PREFIX_PATH="$(python -c 'import sys; print(sys.prefix)')"
python -m pip install --no-build-isolation --no-deps "$CUDNN_DIR"

python - <<'CHECK'
from importlib.metadata import version

import torch
import cutlass.cute as cute
import cudnn
from cudnn import BSA

cute.make_fragment
print("Torch:", torch.__version__, torch.__file__)
print("nvidia-cutlass-dsl:", version("nvidia-cutlass-dsl"))
print("cuDNN Frontend:", version("nvidia-cudnn-frontend"))
print("cuDNN:", cudnn.__file__)
print(BSA.block_sparse_attention_forward, BSA.block_sparse_attention_backward)
print("Installation checks passed.")
CHECK
