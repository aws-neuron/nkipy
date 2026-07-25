import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that start the runtime / require local HF snapshots.",
    )
    parser.addoption(
        "--run-device-gpt-oss",
        action="store_true",
        default=False,
        help="Run TP8 GPT-OSS device tests (Trn2 + local unsloth/gpt-oss-120b-BF16 cache required).",
    )
    parser.addoption(
        "--run-device-qwen3-dense",
        action="store_true",
        default=False,
        help="Run TP8 Qwen3 dense device tests (Trn2 + local Qwen/Qwen3-0.6B cache required).",
    )
    parser.addoption(
        "--run-device-qwen3-moe",
        action="store_true",
        default=False,
        help="Run TP4 Qwen3 MoE device tests (Trn2 + local Qwen/Qwen3-30B-A3B-Thinking-2507 cache required).",
    )
    parser.addoption(
        "--run-device-ep",
        action="store_true",
        default=False,
        help="Run TP8+EP16 device tests (trn2.48xlarge with NEURON_LOGICAL_NC_CONFIG=1 required).",
    )
    parser.addoption(
        "--run-device-dsv4",
        action="store_true",
        default=False,
        help="Run DeepSeek-V4 TP8+EP8 device tests (trn2.48xlarge with NEURON_LOGICAL_NC_CONFIG=1 required).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: integration tests (slow; may require local HuggingFace snapshots).",
    )
    config.addinivalue_line(
        "markers",
        "device_gpt_oss: GPT-OSS device tests (very slow; requires Neuron cores and local HF cache).",
    )
    config.addinivalue_line(
        "markers",
        "device_qwen3_dense: Qwen3 dense device tests (very slow; requires Neuron cores and local HF cache).",
    )
    config.addinivalue_line(
        "markers",
        "device_qwen3_moe: Qwen3 MoE device tests (very slow; requires Neuron cores and local HF cache).",
    )
    config.addinivalue_line(
        "markers",
        "device_ep: EP device tests (very slow; requires 128 Neuron cores with lnc=1 and local HF cache).",
    )
    config.addinivalue_line(
        "markers",
        "device_dsv4: DeepSeek-V4 device tests (very slow; requires 128 Neuron cores with lnc=1 and prepared local weights).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration tests unless explicitly enabled."""
    run_integration = bool(config.getoption("--run-integration"))
    run_device_gpt_oss = bool(config.getoption("--run-device-gpt-oss"))
    run_device_qwen3_dense = bool(config.getoption("--run-device-qwen3-dense"))
    run_device_qwen3_moe = bool(config.getoption("--run-device-qwen3-moe"))
    run_device_ep = bool(config.getoption("--run-device-ep"))
    run_device_dsv4 = bool(config.getoption("--run-device-dsv4"))
    skip_integration = pytest.mark.skip(reason="need --run-integration to run")
    skip_device_gpt_oss = pytest.mark.skip(reason="need --run-device-gpt-oss to run")
    skip_device_qwen3_dense = pytest.mark.skip(
        reason="need --run-device-qwen3-dense to run"
    )
    skip_device_qwen3_moe = pytest.mark.skip(
        reason="need --run-device-qwen3-moe to run"
    )
    skip_device_ep = pytest.mark.skip(reason="need --run-device-ep to run")
    skip_device_dsv4 = pytest.mark.skip(reason="need --run-device-dsv4 to run")
    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "device_gpt_oss" in item.keywords and not run_device_gpt_oss:
            item.add_marker(skip_device_gpt_oss)
        if "device_qwen3_dense" in item.keywords and not run_device_qwen3_dense:
            item.add_marker(skip_device_qwen3_dense)
        if "device_qwen3_moe" in item.keywords and not run_device_qwen3_moe:
            item.add_marker(skip_device_qwen3_moe)
        if "device_ep" in item.keywords and not run_device_ep:
            item.add_marker(skip_device_ep)
        if "device_dsv4" in item.keywords and not run_device_dsv4:
            item.add_marker(skip_device_dsv4)
