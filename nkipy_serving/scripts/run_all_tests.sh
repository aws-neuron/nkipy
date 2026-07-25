#!/bin/bash
# Run all nkipy-serving tests sequentially with zero skipped tests.
#
# Usage:
#   bash scripts/run_all_tests.sh              # run all tests (use cached NEFFs)
#   bash scripts/run_all_tests.sh --clean      # clear compiled kernel caches first
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
cd "$PACKAGE_ROOT"

PYTEST_XML_DIR="${PYTEST_XML_DIR:-/tmp/nkipy_serving_pytest_xml}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/.venv/bin/python}"
DSV4_HF_MODEL_ID="${NKIPY_SERVING_DSV4_HF_MODEL_ID:-${NKIPY_SERVING_HF_MODEL_ID:-}}"
DSV4_TOKENIZER_MODEL_ID="${NKIPY_SERVING_DSV4_TOKENIZER_MODEL_ID:-${NKIPY_SERVING_TOKENIZER_MODEL_ID:-}}"
DSV4_PREPARED_WEIGHT_DIR="${NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR:-}"
mkdir -p "$PYTEST_XML_DIR"

# --- Cache control ---
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning compiled kernel caches..."
    rm -rf /tmp/build /tmp/build_tp8 /tmp/build_tp8_ep16 /tmp/build_tp4 /tmp/build_gpt_oss_bench /tmp/build_dsv4_smoke 2>/dev/null || true
    echo "Done."
fi

# Kill any stale Neuron worker processes from prior runs.
pkill -9 -f spawn_main 2>/dev/null || true
sleep 2

PASS=0
FAIL=0

assert_no_skips() {
    local xml="$1"
    "$PYTHON_BIN" - "$xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
if root.tag == "testsuite":
    suites = [root]
else:
    suites = list(root.iter("testsuite"))
skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
if skipped:
    raise SystemExit(f"{skipped} skipped test(s)")
PY
}

slugify() {
    printf "%s" "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_'
}

run_test() {
    local label="$1"; shift
    local slug
    slug="$(slugify "$label")"
    local xml="$PYTEST_XML_DIR/${slug}.xml"
    echo ""
    echo "======================================================================"
    echo "  $label"
    echo "======================================================================"
    if "$@" --junitxml="$xml" && assert_no_skips "$xml"; then
        echo "  ✓ $label PASSED"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $label FAILED"
        FAIL=$((FAIL + 1))
    fi
    # Clean up stale Neuron processes between device tests.
    pkill -9 -f spawn_main 2>/dev/null || true
    sleep 2
}

echo "Starting all tests at $(date)"

# --- Unit tests (fast, no device) ---
run_test "Unit tests" \
    "$PYTHON_BIN" -m pytest -v --tb=short -m "not integration"

# --- Non-model-specific integration tests ---
run_test "Integration: non-device" \
    "$PYTHON_BIN" -m pytest --run-integration -v --tb=short \
        -m "integration and not device_gpt_oss and not device_qwen3_dense and not device_qwen3_moe and not device_ep and not device_dsv4"

# --- Device tests (require Neuron cores) ---
run_test "Device: TP8 Qwen3 serving" \
    env NKIPY_SERVING_QWEN3_DENSE_READY_TIMEOUT_S="${NKIPY_SERVING_QWEN3_DENSE_READY_TIMEOUT_S:-3600}" \
        "$PYTHON_BIN" -m pytest --run-integration --run-device-qwen3-dense \
        tests/test_tp8_serving.py -v --tb=short

run_test "Device: TP8 reload weights" \
    env NKIPY_SERVING_QWEN3_DENSE_READY_TIMEOUT_S="${NKIPY_SERVING_QWEN3_DENSE_READY_TIMEOUT_S:-3600}" \
        "$PYTHON_BIN" -m pytest --run-integration --run-device-qwen3-dense \
        tests/test_tp8_reload_weights.py -v --tb=short

run_test "Device: Qwen3 MoE TP4" \
    env NKIPY_SERVING_QWEN3_MOE_READY_TIMEOUT_S="${NKIPY_SERVING_QWEN3_MOE_READY_TIMEOUT_S:-3600}" \
        "$PYTHON_BIN" -m pytest --run-integration --run-device-qwen3-moe \
        tests/test_qwen3_moe_tp4_device.py -v --tb=short

run_test "Device: GPT-OSS TP8" \
    env NKIPY_SERVING_GPT_OSS_READY_TIMEOUT_S="${NKIPY_SERVING_GPT_OSS_READY_TIMEOUT_S:-3600}" \
        "$PYTHON_BIN" -m pytest --run-integration --run-device-gpt-oss \
        tests/test_gpt_oss_tp8_device.py -v --tb=short

run_test "Device: GPT-OSS TP8+EP16 (LNC=1)" \
    env NEURON_LOGICAL_NC_CONFIG=1 \
        NKIPY_SERVING_GPT_OSS_READY_TIMEOUT_S="${NKIPY_SERVING_GPT_OSS_READY_TIMEOUT_S:-3600}" \
        "$PYTHON_BIN" -m pytest --run-integration --run-device-ep \
        tests/test_gpt_oss_tp8_ep16_device.py -v --tb=short

run_test "Device: DeepSeek-V4 TP8+EP8+R1 4k (LNC=1)" \
    env NEURON_LOGICAL_NC_CONFIG=1 \
        NKIPY_SERVING_HF_MODEL_ID="$DSV4_HF_MODEL_ID" \
        NKIPY_SERVING_TOKENIZER_MODEL_ID="$DSV4_TOKENIZER_MODEL_ID" \
        NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR="$DSV4_PREPARED_WEIGHT_DIR" \
        NKIPY_SERVING_DSV4_PREPARED_WEIGHT_LOCAL_DIR="${NKIPY_SERVING_DSV4_PREPARED_WEIGHT_LOCAL_DIR:-/tmp/dsv4_prepared_smoke}" \
        NKIPY_SERVING_COLLECTIVE_LOAD_TIMEOUT_S="${NKIPY_SERVING_COLLECTIVE_LOAD_TIMEOUT_S:-7200}" \
        NKIPY_SERVING_TP_WORKER_TIMEOUT_S="${NKIPY_SERVING_TP_WORKER_TIMEOUT_S:-7200}" \
        NKIPY_SERVING_SCHEDULER_READY_TIMEOUT_S="${NKIPY_SERVING_SCHEDULER_READY_TIMEOUT_S:-7200}" \
        NKIPY_SERVING_DSV4_READY_TIMEOUT_S="${NKIPY_SERVING_DSV4_READY_TIMEOUT_S:-7200}" \
        NKIPY_SERVING_DSV4_REQUEST_TIMEOUT_S="${NKIPY_SERVING_DSV4_REQUEST_TIMEOUT_S:-3600}" \
        NKIPY_SERVING_DSV4_DEVICE_CONFIG="${NKIPY_SERVING_DSV4_DEVICE_CONFIG:-tests/runtime.tp8_ep8_r1.deepseek_v4.multi_bucket_4k.test.json}" \
        "$PYTHON_BIN" -m pytest --run-integration --run-device-dsv4 \
        tests/test_deepseek_v4_device.py -v --tb=short

run_test "Device: DeepSeek-V4 bucket writes (LNC=1)" \
    env NEURON_LOGICAL_NC_CONFIG=1 \
        NEURON_RT_VISIBLE_CORES="${NKIPY_SERVING_DSV4_KERNEL_TEST_CORE:-0}" \
        "$PYTHON_BIN" -m pytest --run-integration --run-device-dsv4 \
        tests/test_dsv4_writeswa_bucket_device.py -v --tb=short

echo ""
echo "======================================================================"
echo "  ALL DONE: $PASS passed, $FAIL failed, 0 skipped ($(date))"
echo "======================================================================"
[ "$FAIL" -eq 0 ]
