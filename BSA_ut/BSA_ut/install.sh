#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# When BSA_ut lives inside a cudnn-frontend checkout, install that checkout so
# local kernel edits are actually benchmarked.  The standalone bundle keeps
# the historical nested-clone behavior.  CUDNN_DIR remains an explicit
# override for remote-node layouts.
if [[ -z "${CUDNN_DIR:-}" ]]; then
    if [[ -d "$REPO_ROOT/.git" && -f "$REPO_ROOT/pyproject.toml" && -d "$REPO_ROOT/python/cudnn/block_sparse_attention" ]]; then
        CUDNN_DIR="$REPO_ROOT"
    else
        CUDNN_DIR="$SCRIPT_DIR/cudnn-frontend"
    fi
fi
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
python -m pip install --force-reinstall --no-build-isolation --no-deps "$CUDNN_DIR"

python - <<'CHECK'
from importlib.metadata import version
from inspect import signature

import torch
import triton
import cutlass.cute as cute
import cudnn
from cudnn import BSA

assert hasattr(cute, "make_fragment_like")
assert "block_causal" in signature(
    BSA.block_sparse_attention_backward
).parameters, "installed cuDNN Frontend does not contain the local BSA optimization"
assert "backward_backend" in signature(
    BSA.block_sparse_attention_backward
).parameters, "installed cuDNN Frontend does not contain the production split backend"
print("Torch:", torch.__version__, torch.__file__)
print("Triton:", triton.__version__)
print("nvidia-cutlass-dsl:", version("nvidia-cutlass-dsl"))
print("cuDNN Frontend:", version("nvidia-cudnn-frontend"))
print("cuDNN:", cudnn.__file__)
print(BSA.block_sparse_attention_forward, BSA.block_sparse_attention_backward)
print("Installation checks passed.")
CHECK
