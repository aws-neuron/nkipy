import pytest

from nkipy_serving.server_args import ServerArgs


def test_server_args_cli_contracts():
    args = ServerArgs.from_cli(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "31000",
            "--log-level",
            "warning",
            "--log-level-http",
            "error",
            "--config",
            "/tmp/runtime.json",
            "--max-model-len",
            "8192",
            "--device-offset",
            "8",
            "--workers",
            "1",
            "--access-log",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 31000
    assert args.log_level == "warning"
    assert args.log_level_http == "error"
    assert args.config_path == "/tmp/runtime.json"
    assert args.max_model_len == 8192
    assert args.device_offset == 8
    assert args.workers == 1
    assert args.access_log is True

    args = ServerArgs.from_cli(["--max-model-len", "8192", "--device-offset", "16"])
    assert args.max_model_len == 8192
    assert args.device_offset == 16


def test_server_args_fail_fast_contracts():
    cases = [
        (ServerArgs(host=""), "host must be non-empty"),
        (ServerArgs(port=70000), "port must be in"),
        (ServerArgs(workers=2), "single uvicorn worker"),
        (ServerArgs(device_offset=-1), "device_offset must be >= 0"),
    ]
    for args, match in cases:
        with pytest.raises(RuntimeError, match=match):
            args.check_server_args()
