# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kernel_agent generator module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nkipy.tools.kernel_agent.generator import (
    _build_inputs,
    generate_kernel,
)

# ---------------------------------------------------------------------------
# _build_inputs
# ---------------------------------------------------------------------------


class TestBuildInputs:
    def test_float_input(self):
        specs = {"x": {"shape": [2, 3], "dtype": "float32"}}
        result = _build_inputs(specs)
        assert "x" in result
        assert result["x"].shape == (2, 3)
        assert result["x"].dtype == np.float32

    def test_integer_input(self):
        specs = {"idx": {"shape": [4], "dtype": "int32"}}
        result = _build_inputs(specs)
        assert result["idx"].shape == (4,)
        assert result["idx"].dtype == np.int32

    def test_multiple_inputs(self):
        specs = {
            "a": {"shape": [8, 8], "dtype": "float16"},
            "b": {"shape": [8, 8], "dtype": "float16"},
        }
        result = _build_inputs(specs)
        assert len(result) == 2
        assert result["a"].shape == (8, 8)
        assert result["b"].dtype == np.float16

    def test_missing_shape_raises(self):
        specs = {"x": {"dtype": "float32"}}
        with pytest.raises(KeyError):
            _build_inputs(specs)

    def test_missing_dtype_raises(self):
        specs = {"x": {"shape": [2]}}
        with pytest.raises(KeyError):
            _build_inputs(specs)

    def test_bfloat16_input(self):
        specs = {"x": {"shape": [4, 4], "dtype": "bfloat16"}}
        result = _build_inputs(specs)
        assert result["x"].shape == (4, 4)
        assert result["x"].dtype == np.dtype("bfloat16")

    def test_invalid_dtype_raises(self):
        specs = {"x": {"shape": [2], "dtype": "not_a_real_dtype"}}
        with pytest.raises(TypeError):
            _build_inputs(specs)

    def test_empty_specs(self):
        assert _build_inputs({}) == {}


# ---------------------------------------------------------------------------
# generate_kernel — tool use path
# ---------------------------------------------------------------------------


class TestGenerateKernelToolUse:
    """Test the preferred toolUse response path."""

    def _mock_converse_tool_use(self, **kwargs):
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "test-id",
                                "name": "submit_kernel",
                                "input": {
                                    "name": "vec_add",
                                    "code": "def vec_add(a, b):\n    return a + b",
                                    "inputs": {
                                        "a": {"shape": [4], "dtype": "float32"},
                                        "b": {"shape": [4], "dtype": "float32"},
                                    },
                                },
                            }
                        }
                    ]
                }
            }
        }

    @patch("nkipy.tools.kernel_agent.generator.boto3")
    @patch("nkipy.tools.kernel_agent.generator.load_prompt", return_value="sys")
    def test_tool_use_path(self, _mock_prompt, mock_boto3):
        mock_client = MagicMock()
        mock_client.converse.side_effect = self._mock_converse_tool_use
        mock_boto3.client.return_value = mock_client

        name, code, inputs = generate_kernel("add two vectors")

        assert name == "vec_add"
        assert "def vec_add" in code
        assert "a" in inputs and "b" in inputs
        assert inputs["a"].shape == (4,)

        # Verify toolConfig was passed
        call_kwargs = mock_client.converse.call_args.kwargs
        assert "toolConfig" in call_kwargs
        assert (
            call_kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "submit_kernel"
        )


# ---------------------------------------------------------------------------
# generate_kernel — missing toolUse raises
# ---------------------------------------------------------------------------


class TestGenerateKernelNoToolUse:
    """Verify we raise when the model returns text instead of toolUse."""

    @patch("nkipy.tools.kernel_agent.generator.boto3")
    @patch("nkipy.tools.kernel_agent.generator.load_prompt", return_value="sys")
    def test_raises_on_text_only_response(self, _mock_prompt, mock_boto3):
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Here is your kernel..."}]}}
        }
        mock_boto3.client.return_value = mock_client

        with pytest.raises(ValueError, match="did not return a toolUse"):
            generate_kernel("multiply by 2")


# ---------------------------------------------------------------------------
# generate_kernel — constrained mode
# ---------------------------------------------------------------------------


class TestGenerateKernelConstrained:
    @patch("nkipy.tools.kernel_agent.generator.boto3")
    @patch("nkipy.tools.kernel_agent.generator.load_prompt", return_value="sys")
    def test_constrained_passes_ops(self, _mock_prompt, mock_boto3):
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "id",
                                "name": "submit_kernel",
                                "input": {
                                    "name": "k",
                                    "code": "def k(x): return x",
                                    "inputs": {},
                                },
                            }
                        }
                    ]
                }
            }
        }
        mock_boto3.client.return_value = mock_client

        generate_kernel("identity", constrained=True, allowed_ops=["np.add"])

        call_kwargs = mock_client.converse.call_args.kwargs
        user_msg = call_kwargs["messages"][0]["content"][0]["text"]
        assert "np.add" in user_msg
        assert "ALLOWED OPS" in user_msg
