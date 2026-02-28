# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kernel_agent sweep module."""

import json
from unittest.mock import patch

import numpy as np
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from nkipy.tools.kernel_agent.executor import ExecutionResult, StageResult
from nkipy.tools.kernel_agent.sweep import SweepRecord, run_rerun, run_sweep

# Patch targets: lazy imports in _run_and_record / run_sweep / run_rerun resolve from source modules
_GEN = "nkipy.tools.kernel_agent.generator.generate_kernel"
_COMPILE = "nkipy.tools.kernel_agent.generator.compile_code"
_BUILD = "nkipy.tools.kernel_agent.generator.build_inputs"
_RUN = "nkipy.tools.kernel_agent.executor.run_kernel"
_COMPARE = "nkipy.tools.kernel_agent.executor.compare_outputs"
_PROMPTS = "nkipy.tools.kernel_agent.sweep.get_kernel_prompts"


def _make_passing_result():
    """Create a mock ExecutionResult that passes all stages."""
    return ExecutionResult(
        numpy=StageResult(success=True, output=np.array([1.0, 2.0])),
        compile=StageResult(success=True),
        hardware=StageResult(success=True, output=np.array([1.0, 2.0])),
    )


def _read_jsonl(path):
    """Read all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_source_jsonl(path, records):
    """Write a list of record dicts to a JSONL file."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_source_record(*, passed=True, code="def k(x): return x", iteration=1):
    """Create a source sweep record dict."""
    return {
        "iteration": iteration,
        "timestamp": "2025-01-01T00:00:00",
        "prompt": "test prompt",
        "model_id": "test-model",
        "kernel_name": "k",
        "generated_code": code,
        "input_specs": {"x": {"shape": [4], "dtype": "float32"}},
        "passed": passed,
        "failure_stage": None if passed else "compile",
        "error_message": None if passed else "compile failed",
        "duration_seconds": 1.0,
    }


# ---------------------------------------------------------------------------
# Sweep tests
# ---------------------------------------------------------------------------


class TestSweep:
    @patch(_PROMPTS, return_value=["test prompt"])
    @patch(_GEN)
    def test_generation_failure_is_logged(self, mock_generate, mock_prompts, tmp_path):
        """ValueError from generate_kernel -> record written to JSONL."""
        mock_generate.side_effect = ValueError("Model returned text")

        log_path = run_sweep(
            max_iterations=1,
            output_dir=str(tmp_path),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["failure_stage"] == "generation"
        assert "Model returned text" in records[0]["error_message"]
        assert records[0]["passed"] is False

    @patch(_PROMPTS, return_value=["test prompt"])
    @patch(_COMPARE)
    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_GEN)
    def test_compile_code_failure_is_logged(
        self,
        mock_generate,
        mock_compile,
        mock_run,
        mock_compare,
        mock_prompts,
        tmp_path,
    ):
        """compile_code returns None -> record written to JSONL."""
        mock_generate.return_value = (
            "test_kernel",
            "def test_kernel(x): return x",
            {"x": np.array([1.0])},
        )
        mock_compile.return_value = None

        log_path = run_sweep(
            max_iterations=1,
            output_dir=str(tmp_path),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["failure_stage"] == "compile_code"
        assert records[0]["passed"] is False

    @patch(_PROMPTS, return_value=["test prompt"])
    @patch(_GEN)
    def test_bedrock_client_error_stops_sweep(
        self, mock_generate, mock_prompts, tmp_path
    ):
        """ClientError raised -> sweep stops after 1 iteration, record still written."""
        mock_generate.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "Converse",
        )

        with pytest.raises(ClientError):
            run_sweep(
                max_iterations=5,
                output_dir=str(tmp_path),
                timeout=0,
            )

        # Find the JSONL file
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        records = _read_jsonl(jsonl_files[0])
        assert len(records) == 1
        assert records[0]["failure_stage"] == "generation"
        assert "Access denied" in records[0]["error_message"]

    @patch(_PROMPTS, return_value=["test prompt"])
    @patch(_GEN)
    def test_bedrock_no_credentials_stops_sweep(
        self, mock_generate, mock_prompts, tmp_path
    ):
        """NoCredentialsError -> sweep stops, record still written."""
        mock_generate.side_effect = NoCredentialsError()

        with pytest.raises(NoCredentialsError):
            run_sweep(
                max_iterations=5,
                output_dir=str(tmp_path),
                timeout=0,
            )

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        records = _read_jsonl(jsonl_files[0])
        assert len(records) == 1
        assert records[0]["failure_stage"] == "generation"

    @patch(_PROMPTS, return_value=["test prompt"])
    @patch(_COMPARE)
    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_GEN)
    def test_value_error_continues_sweep(
        self,
        mock_generate,
        mock_compile,
        mock_run,
        mock_compare,
        mock_prompts,
        tmp_path,
    ):
        """ValueError on iter 1, success on iter 2 -> 2 records in JSONL."""
        passing_result = _make_passing_result()

        mock_generate.side_effect = [
            ValueError("Model returned text"),
            ("good_kernel", "def good_kernel(x): return x", {"x": np.array([1.0])}),
        ]
        mock_compile.return_value = lambda x: x
        mock_run.return_value = passing_result
        mock_compare.return_value = {
            "max_diff": 0.0,
            "allclose": True,
            "shapes_match": True,
        }

        log_path = run_sweep(
            max_iterations=2,
            output_dir=str(tmp_path),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 2
        assert records[0]["failure_stage"] == "generation"
        assert "Model returned text" in records[0]["error_message"]
        assert records[0]["passed"] is False
        assert records[1]["kernel_name"] == "good_kernel"
        assert records[1]["passed"] is True
        assert records[1]["allclose"] is True

    @patch(_PROMPTS, return_value=["test prompt"])
    @patch(_COMPARE)
    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_GEN)
    def test_compile_code_exception_is_logged(
        self,
        mock_generate,
        mock_compile,
        mock_run,
        mock_compare,
        mock_prompts,
        tmp_path,
    ):
        """compile_code raising SyntaxError -> caught by catch-all, sweep continues."""
        mock_generate.side_effect = [
            ("k1", "bad code", {"x": np.array([1.0])}),
            ("k2", "def k2(x): return x", {"x": np.array([1.0])}),
        ]
        mock_compile.side_effect = [SyntaxError("invalid syntax"), lambda x: x]
        mock_run.return_value = _make_passing_result()
        mock_compare.return_value = {
            "max_diff": 0.0,
            "allclose": True,
            "shapes_match": True,
        }

        log_path = run_sweep(
            max_iterations=2,
            output_dir=str(tmp_path),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 2
        assert records[0]["failure_stage"] == "unexpected"
        assert "invalid syntax" in records[0]["error_message"]
        assert records[0]["passed"] is False
        assert records[1]["passed"] is True


# ---------------------------------------------------------------------------
# Rerun tests
# ---------------------------------------------------------------------------


class TestRerun:
    @patch(_COMPARE)
    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_BUILD)
    def test_basic(self, mock_build, mock_compile, mock_run, mock_compare, tmp_path):
        """Single valid record -> rerun passes, output has rerun_source set."""
        source = tmp_path / "source.jsonl"
        _write_source_jsonl(source, [_make_source_record()])

        mock_build.return_value = {"x": np.zeros(4, dtype=np.float32)}
        mock_compile.return_value = lambda x: x
        mock_run.return_value = _make_passing_result()
        mock_compare.return_value = {
            "max_diff": 0.0,
            "allclose": True,
            "shapes_match": True,
        }

        out_dir = tmp_path / "output"
        log_path = run_rerun(
            source_path=str(source),
            output_dir=str(out_dir),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["passed"] is True
        assert records[0]["rerun_source"] == str(source)
        assert records[0]["max_diff"] == 0.0
        assert records[0]["allclose"] is True
        assert records[0]["shapes_match"] is True
        assert records[0]["numpy_success"] is True
        assert records[0]["compile_success"] is True
        assert records[0]["hardware_success"] is True
        assert records[0]["kernel_name"] == "k"

    @patch(_COMPARE)
    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_BUILD)
    def test_skips_no_code(
        self, mock_build, mock_compile, mock_run, mock_compare, tmp_path
    ):
        """Records without generated_code are skipped."""
        source = tmp_path / "source.jsonl"
        no_code_rec = _make_source_record()
        no_code_rec["generated_code"] = ""
        valid_rec = _make_source_record(iteration=2)

        _write_source_jsonl(source, [no_code_rec, valid_rec])

        mock_build.return_value = {"x": np.zeros(4, dtype=np.float32)}
        mock_compile.return_value = lambda x: x
        mock_run.return_value = _make_passing_result()
        mock_compare.return_value = {
            "max_diff": 0.0,
            "allclose": True,
            "shapes_match": True,
        }

        out_dir = tmp_path / "output"
        log_path = run_rerun(
            source_path=str(source),
            output_dir=str(out_dir),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["iteration"] == 1
        assert records[0]["kernel_name"] == "k"
        assert records[0]["passed"] is True

    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_BUILD)
    def test_compile_failure(self, mock_build, mock_compile, mock_run, tmp_path):
        """compile_code returns None -> failure logged."""
        source = tmp_path / "source.jsonl"
        _write_source_jsonl(source, [_make_source_record()])

        mock_build.return_value = {"x": np.zeros(4, dtype=np.float32)}
        mock_compile.return_value = None

        out_dir = tmp_path / "output"
        log_path = run_rerun(
            source_path=str(source),
            output_dir=str(out_dir),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["failure_stage"] == "compile_code"
        assert records[0]["passed"] is False

    def test_file_not_found(self, tmp_path):
        """Nonexistent path -> FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            run_rerun(
                source_path=str(tmp_path / "nonexistent.jsonl"),
                output_dir=str(tmp_path / "output"),
            )

    def test_rejects_non_jsonl(self, tmp_path):
        """File exists but wrong extension -> ValueError."""
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b,c\n")

        with pytest.raises(ValueError, match=r"\.jsonl"):
            run_rerun(
                source_path=str(bad_file),
                output_dir=str(tmp_path / "output"),
            )

    def test_no_eligible_raises(self, tmp_path):
        """All records lack code -> ValueError."""
        source = tmp_path / "empty.jsonl"
        rec = _make_source_record()
        rec["generated_code"] = ""
        _write_source_jsonl(source, [rec])

        with pytest.raises(ValueError, match="No eligible records"):
            run_rerun(
                source_path=str(source),
                output_dir=str(tmp_path / "output"),
            )

    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_BUILD)
    def test_numpy_failure(self, mock_build, mock_compile, mock_run, tmp_path):
        """run_kernel returns numpy failure -> record has failure_stage='numpy'."""
        source = tmp_path / "source.jsonl"
        _write_source_jsonl(source, [_make_source_record()])

        mock_build.return_value = {"x": np.zeros(4, dtype=np.float32)}
        mock_compile.return_value = lambda x: x
        mock_run.return_value = ExecutionResult(
            numpy=StageResult(success=False, error="numpy boom"),
            compile=None,
            hardware=None,
        )

        out_dir = tmp_path / "output"
        log_path = run_rerun(
            source_path=str(source),
            output_dir=str(out_dir),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["failure_stage"] == "numpy"
        assert records[0]["error_message"] == "numpy boom"
        assert records[0]["passed"] is False

    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_BUILD)
    def test_hardware_failure(self, mock_build, mock_compile, mock_run, tmp_path):
        """numpy+compile pass, hardware fails -> record has failure_stage='hardware'."""
        source = tmp_path / "source.jsonl"
        _write_source_jsonl(source, [_make_source_record()])

        mock_build.return_value = {"x": np.zeros(4, dtype=np.float32)}
        mock_compile.return_value = lambda x: x
        mock_run.return_value = ExecutionResult(
            numpy=StageResult(success=True, output=np.array([1.0])),
            compile=StageResult(success=True),
            hardware=StageResult(success=False, error="hardware boom"),
        )

        out_dir = tmp_path / "output"
        log_path = run_rerun(
            source_path=str(source),
            output_dir=str(out_dir),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["failure_stage"] == "hardware"
        assert records[0]["error_message"] == "hardware boom"
        assert records[0]["passed"] is False

    @patch(_RUN)
    @patch(_COMPILE)
    @patch(_BUILD)
    def test_unexpected_exception_is_logged(
        self, mock_build, mock_compile, mock_run, tmp_path
    ):
        """RuntimeError from compile_code -> failure_stage='unexpected', rerun continues."""
        source = tmp_path / "source.jsonl"
        _write_source_jsonl(
            source, [_make_source_record(), _make_source_record(iteration=2)]
        )

        mock_build.return_value = {"x": np.zeros(4, dtype=np.float32)}
        mock_compile.side_effect = [
            RuntimeError("something broke"),
            lambda x: x,
        ]
        mock_run.return_value = _make_passing_result()

        out_dir = tmp_path / "output"
        log_path = run_rerun(
            source_path=str(source),
            output_dir=str(out_dir),
            timeout=0,
        )

        records = _read_jsonl(log_path)
        assert len(records) == 2
        assert records[0]["failure_stage"] == "unexpected"
        assert "something broke" in records[0]["error_message"]
        assert records[0]["passed"] is False
        assert records[1]["passed"] is True

    def test_cli_args(self):
        """CLI arg parsing works correctly for rerun subcommand."""
        from nkipy.tools.kernel_agent.__main__ import main

        with patch("nkipy.tools.kernel_agent.sweep.run_rerun") as mock_rerun:
            mock_rerun.return_value = "fake_path"
            main(["rerun", "/tmp/test.jsonl", "--no-hardware"])

            mock_rerun.assert_called_once_with(
                source_path="/tmp/test.jsonl",
                run_hardware=False,
                output_dir="sweep_results",
                timeout=120,
                summary_interval=10,
            )
