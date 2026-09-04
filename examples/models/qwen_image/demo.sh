#!/bin/bash
# End-to-end demo for Qwen-Image (20B MMDiT) on Trainium.
# Usage: bash demo.sh ["your prompt"]
#
# Runs the CPU correctness tests and generates one image at 512px end-to-end on
# device: the denoiser (fused CFG + FlowMatchEuler sampling loop), the Qwen2.5
# text encoder, and the VAE decoder all run on the TP cores.
#
# The expensive one-time work is cached, so reruns are fast:
#   * The ~40 GB model is downloaded once by the diffusers pipeline into the HF
#     cache; the driver extracts + shards all weights in-memory (no repack step).
#   * Compiled kernels are kept in build/ build_te/ build_vae/ and reused across
#     runs (content-hash keyed, so kernel edits recompile automatically).
#
# Env knobs: TP HEIGHT WIDTH STEPS OUTPUT (see below); CLEAN=1 forces a kernel
# recompile; SKIP_TESTS=1 skips the CPU tests.
#
# Notes:
#   * The 20B model does not fit on one core (~40 GB bf16 > 24 GB/core), so it is
#     tensor-parallel over $TP cores via torchrun. TP is required; TP=4 fits (and
#     is the max — the text encoder caps TP at num_kv_heads=4).
#   * 512px (grid 32x32) is the working resolution; native 1024px currently
#     exceeds the compiler's 2 GB HLO-proto limit (see README.md).
set -e

MODEL="Qwen/Qwen-Image"
TP=${TP:-4}                       # tensor-parallel degree (override: TP=8 bash demo.sh)
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-512}
STEPS=${STEPS:-20}
PROMPT="${1:-a photorealistic coffee shop storefront at golden hour, a wooden chalkboard sign reading 'Qwen Cafe', warm cinematic lighting, highly detailed}"
OUTPUT="${OUTPUT:-output.png}"

echo "=========================================="
echo "Qwen-Image (20B MMDiT) on Trainium — demo"
echo "  TP=$TP  ${WIDTH}x${HEIGHT}  steps=$STEPS"
echo "=========================================="

# Step 1: compilation cache. Kept across runs so reruns skip the neuronx-cc
# compiles (denoiser ~33s, text encoder, VAE). The caches are content-hash
# keyed, so editing a kernel recompiles automatically — no stale-cache risk.
# Force a clean rebuild with CLEAN=1.
echo ""
if [ "${CLEAN:-0}" = "1" ]; then
    echo "[1/3] CLEAN=1 -> clearing compilation caches (build/ build_te/ build_vae/)..."
    rm -rf build/ build_te/ build_vae/ 2>/dev/null || true
    echo "OK caches cleared"
else
    echo "[1/3] Reusing compilation cache (set CLEAN=1 to force a recompile)."
fi

# Step 2: CPU correctness tests (kernels validated vs diffusers). Skip on
# reruns with SKIP_TESTS=1.
echo ""
if [ "${SKIP_TESTS:-0}" = "1" ]; then
    echo "[2/3] SKIP_TESTS=1 -> skipping CPU correctness tests."
else
    echo "[2/3] Running CPU correctness tests..."
    python -m pytest tests/ -q
fi

# Step 3: generate one image end-to-end (TP; denoiser + text encoder + VAE on
# device). The driver downloads the model (first run) and extracts + shards all
# weights in-memory from the host pipeline.
echo ""
echo "[3/3] Generating an image (TP=$TP)..."
echo "=========================================="
torchrun --nproc-per-node "$TP" qwen_image.py "$PROMPT" \
    --model "$MODEL" \
    --height "$HEIGHT" --width "$WIDTH" --steps "$STEPS" \
    --output "$OUTPUT"

echo ""
echo "=========================================="
echo "OK Demo complete! Image written to $OUTPUT"
echo "=========================================="
