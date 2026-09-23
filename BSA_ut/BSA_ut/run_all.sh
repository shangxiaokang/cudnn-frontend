#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || "$1" == --help ]]; then
    echo 'Usage: bash run_all.sh CLOCK_MHZ'
    echo 'Example: GPU=0 bash run_all.sh 2032'
    echo 'Optional: WARMUP=5 RUNS=10 BSA_CAUSAL_BWD_BACKEND=blk64|split QMAJOR_BLOCK_N=32|64 BUCKET_SIZE_BLOCKS=...'
    echo 'Backend default when unset: bsa-causal=flex, production=split; run bsa-causal directly to select flex explicitly.'
    [[ "${1:-}" == --help ]] && exit 0
    exit 2
fi
[[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo 'CLOCK_MHZ must be a positive integer' >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CLOCK_MHZ="$1"
GPU="${GPU:-0}"
WARMUP="${WARMUP:-5}"
RUNS="${RUNS:-10}"
BSA_CAUSAL_BWD_BACKEND="${BSA_CAUSAL_BWD_BACKEND:-}"
QMAJOR_BLOCK_N="${QMAJOR_BLOCK_N:-32}"
BUCKET_SIZE_BLOCKS="${BUCKET_SIZE_BLOCKS:-}"
BSA_PYTHON="${BSA_PYTHON:-$SCRIPT_DIR/.venv/bsa/bin/python}"
CAUSAL_PYTHON="${CAUSAL_PYTHON:-$SCRIPT_DIR/.venv/causal/bin/python}"
if [[ "${PRODUCTION_BWD_BACKEND+x}" == x ]]; then
    echo "PRODUCTION_BWD_BACKEND was removed; use BSA_CAUSAL_BWD_BACKEND=blk64 for the former fused value, or BSA_CAUSAL_BWD_BACKEND=split" >&2
    exit 2
fi
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/results/locked_$(date +%Y%m%d_%H%M%S)_$$}"
[[ -x "$BSA_PYTHON" && -x "$CAUSAL_PYTHON" ]] || { echo 'Both Python environments must be installed first' >&2; exit 2; }
mkdir -p "$(dirname -- "$LOG_DIR")"
mkdir "$LOG_DIR"

locked=0
monitor_pid=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ "$locked" == 1 ]]; then
        if ! nvidia-smi -i "$GPU" -rgc > "$LOG_DIR/unlock.log" 2>&1; then
            cat "$LOG_DIR/unlock.log" >&2
            echo "Failed to reset clocks; run: nvidia-smi -i $GPU -rgc" >&2
            status=1
        fi
    fi
    echo "Exit status: $status; logs: $LOG_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export CUDA_VISIBLE_DEVICES="$GPU"
if [[ -n "$BSA_CAUSAL_BWD_BACKEND" && "$BSA_CAUSAL_BWD_BACKEND" != "blk64" && "$BSA_CAUSAL_BWD_BACKEND" != "flex" && "$BSA_CAUSAL_BWD_BACKEND" != "split" ]]; then
    echo "BSA_CAUSAL_BWD_BACKEND must be blk64, flex, or split" >&2
    exit 2
fi
if [[ "$BSA_CAUSAL_BWD_BACKEND" == flex ]]; then
    echo "run_all.sh includes production, whose arbitrary mask does not support flex; use blk64/split or run the bsa-causal flex command directly" >&2
    exit 2
fi
if [[ "$QMAJOR_BLOCK_N" != "32" && "$QMAJOR_BLOCK_N" != "64" ]]; then
    echo "QMAJOR_BLOCK_N must be 32 or 64" >&2
    exit 2
fi
BACKEND_CONFIG="${BSA_CAUSAL_BWD_BACKEND:-case-defaults(bsa-causal=flex,production=split)}"
echo "GPU=$GPU CLOCK_MHZ=$CLOCK_MHZ WARMUP=$WARMUP RUNS=$RUNS BSA_CAUSAL_BWD_BACKEND=$BACKEND_CONFIG QMAJOR_BLOCK_N=$QMAJOR_BLOCK_N BUCKET_SIZE_BLOCKS=${BUCKET_SIZE_BLOCKS:-auto}" | tee "$LOG_DIR/config.log"
echo "BSA_PYTHON=$BSA_PYTHON CAUSAL_PYTHON=$CAUSAL_PYTHON" | tee -a "$LOG_DIR/config.log"
nvidia-smi -i "$GPU" -q > "$LOG_DIR/gpu_before.log"
# A lock failure stops the run before any benchmark is launched.
if nvidia-smi -i "$GPU" -lgc "$CLOCK_MHZ,$CLOCK_MHZ" > "$LOG_DIR/lock.log" 2>&1; then
    locked=1
else
    status=$?
    cat "$LOG_DIR/lock.log" >&2
    exit "$status"
fi
cat "$LOG_DIR/lock.log"
nvidia-smi -i "$GPU" \
    --query-gpu=timestamp,name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu \
    --format=csv -lms 500 > "$LOG_DIR/clocks.csv" 2> "$LOG_DIR/monitor.log" &
monitor_pid=$!

for case in causal bsa-causal production; do
    if [[ "$case" == causal ]]; then
        case_python="$CAUSAL_PYTHON"
    else
        case_python="$BSA_PYTHON"
    fi
    case_args=()
    if [[ "$case" != causal ]]; then
        effective_backend="$BSA_CAUSAL_BWD_BACKEND"
        if [[ -z "$effective_backend" ]]; then
            if [[ "$case" == bsa-causal ]]; then
                effective_backend=flex
            else
                effective_backend=split
            fi
        fi
        case_args+=(--bsa-causal-bwd-backend "$effective_backend")
        if [[ "$effective_backend" == split ]]; then
            case_args+=(--qmajor-block-n "$QMAJOR_BLOCK_N")
        fi
        if [[ -n "$BUCKET_SIZE_BLOCKS" && "$effective_backend" != flex ]]; then
            case_args+=(--bucket-size-blocks "$BUCKET_SIZE_BLOCKS")
        fi
    fi
    echo "Running $case"
    "$case_python" -u benchmark.py --case "$case" --clock-mhz "$CLOCK_MHZ" --warmup "$WARMUP" --runs "$RUNS" "${case_args[@]}" 2>&1 | tee "$LOG_DIR/$case.log"
done
if ! kill -0 "$monitor_pid" 2>/dev/null; then
    cat "$LOG_DIR/monitor.log" >&2
    echo 'Clock monitoring stopped unexpectedly; check clocks.csv before using the results' >&2
    exit 1
fi
nvidia-smi -i "$GPU" -q > "$LOG_DIR/gpu_after.log"
