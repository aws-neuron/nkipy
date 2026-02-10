# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
from logging.config import dictConfig

from vllm import __version__ as __version__  # noqa: F401
from vllm.logger import DEFAULT_LOGGING_CONFIG


def register():
    """Register NKIPy platform."""
    return "vllm_nkipy.platform.NKIpyPlatform"


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config = {**DEFAULT_LOGGING_CONFIG}

    # Copy the vLLM logging configurations (use dict() to avoid mutating originals)
    config["formatters"]["vllm_nkipy"] = dict(
        DEFAULT_LOGGING_CONFIG["formatters"]["vllm"]
    )

    handler_config = dict(DEFAULT_LOGGING_CONFIG["handlers"]["vllm"])
    handler_config["formatter"] = "vllm_nkipy"
    config["handlers"]["vllm_nkipy"] = handler_config

    logger_config = dict(DEFAULT_LOGGING_CONFIG["loggers"]["vllm"])
    logger_config["handlers"] = ["vllm_nkipy"]
    config["loggers"]["vllm_nkipy"] = logger_config

    # Also configure torch_to_nkipy logger so logs from NKIPy backend are visible
    torch_to_nkipy_logger_config = dict(DEFAULT_LOGGING_CONFIG["loggers"]["vllm"])
    torch_to_nkipy_logger_config["handlers"] = ["vllm_nkipy"]
    config["loggers"]["torch_to_nkipy"] = torch_to_nkipy_logger_config

    dictConfig(config)


_init_logging()
