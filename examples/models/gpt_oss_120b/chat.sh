#!/bin/bash
# Run gpt-oss-120b generation on Trainium (TRN2).
#
# Parallelism layout (see parallel_state.py):
#   world_size = TP_SIZE * DP_SIZE
#   PREFILL_EP_SIZE  expert-parallel groups used during prefill
#   DP_SIZE          data-parallel == decode expert-parallel groups
#
# Weights must be presharded for the chosen TP_SIZE first; see README.md.
set -e

export OMP_NUM_THREADS=1

# ---- Parallelism config (override via env) ----
# Defaults are the layout verified on a single trn2.48xlarge (64 ranks, dp8/ep8).
export TP_SIZE="${TP_SIZE:-8}"
export DP_SIZE="${DP_SIZE:-8}"
export PREFILL_EP_SIZE="${PREFILL_EP_SIZE:-8}"

# Base checkpoint path; the runner appends "-TP${TP_SIZE}".
CHECKPOINT="${CHECKPOINT:-./gpt-oss-120b-bf16}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Activate the repo venv so python, neuronx-cc, and torchrun all resolve there
# (a bare invocation would use the system python, which reads a stale ~/.local
# user-site). Set VENV to override, or VENV= to use the active environment.
VENV="${VENV-$( git -C "$SCRIPT_DIR" rev-parse --show-toplevel )/.venv}"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# Kill any stale run holding the rendezvous port.
lsof -ti:29501 | xargs -r kill -9 || true

torchrun \
    --nproc-per-node=$((TP_SIZE * DP_SIZE)) \
    --master-port=29501 \
    "$SCRIPT_DIR/chat.py" \
    --tp_size="$TP_SIZE" \
    --prefill_ep_size="$PREFILL_EP_SIZE" \
    --checkpoint="$CHECKPOINT" \
    "$@" 2>&1 | tee chat.log
