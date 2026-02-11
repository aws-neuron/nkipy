# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for compiler wrapper functions"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from nkipy.core.compile import (
    CompilationConfig,
    CompilationTarget,
    Compiler,
    compile_to_neff,
    lower_to_nki,
    trace,
)


class TestCompilerWrappers(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test artifacts
        self.test_dir = Path(tempfile.mkdtemp())

        @trace
        def simple_add(x, y):
            return np.add(x, y)

        self.traced_kernel = simple_add

        # Create some sample input tensors
        self.input_x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.input_y = np.array([[5, 6], [7, 8]], dtype=np.float32)

        # Call the kernel to ensure it's traced
        _ = self.traced_kernel.specialize(self.input_x, self.input_y)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_compile_to_neff_basic(self):
        """Test basic NEFF compilation without artifacts"""
        output_dir = self.test_dir / "basic_test"
        output_dir.mkdir()

        neff_path = compile_to_neff(
            self.traced_kernel,
            output_dir=str(output_dir),
            target=CompilationTarget.DEFAULT,
        )

        neff_path = Path(neff_path)

        # Verify the NEFF file was created
        self.assertTrue(neff_path.exists())
        self.assertTrue(neff_path.name.endswith(".neff"))

        # Verify file size is non-zero
        self.assertGreater(neff_path.stat().st_size, 0)

    def test_compile_to_neff_with_artifacts(self):
        """Test NEFF compilation with artifact preservation"""
        output_dir = self.test_dir / "artifacts_test"

        neff_path = compile_to_neff(
            self.traced_kernel,
            output_dir=str(output_dir),
            target=CompilationTarget.DEFAULT,
            save_artifacts=True,
        )
        neff_path = Path(neff_path)

        # Verify expected artifact files exist
        self.assertTrue(output_dir.exists())
        self.assertTrue(neff_path.exists())

        # Verify files have content
        self.assertGreater(neff_path.stat().st_size, 0)

    def test_compile_to_neff_different_targets(self):
        """Test compilation for different targets"""
        for target in CompilationTarget:
            with self.subTest(target=target.value):
                output_dir = self.test_dir / f"target_{target.value}"

                neff_path = compile_to_neff(
                    self.traced_kernel, output_dir=str(output_dir), target=target
                )

                neff_path = Path(neff_path)

                # Verify NEFF file exists and has content
                self.assertTrue(neff_path.exists())
                self.assertGreater(neff_path.stat().st_size, 0)

    def test_lower_to_nki_basic(self):
        """Test basic NKI lowering"""
        nki_code = lower_to_nki(self.traced_kernel)

        # Verify NKI content
        self.assertIsInstance(nki_code, str)
        self.assertGreater(len(nki_code), 0)
        self.assertIn("def", nki_code)  # NKI should contain function definition

    def test_lower_to_nki_with_artifacts(self):
        """Test NKI lowering with artifact preservation"""
        output_dir = self.test_dir / "nki_artifacts"

        nki_code = lower_to_nki(self.traced_kernel, output_dir=str(output_dir))

        # Verify artifacts were saved
        self.assertTrue(output_dir.exists())
        self.assertTrue((output_dir / "nki.py").exists())

    def test_compile_with_invalid_output_dir(self):
        """Test compilation with invalid output directory"""
        invalid_dir = "/nonexistent/path/that/should/not/exist"

        with self.assertRaises(Exception):
            compile_to_neff(self.traced_kernel, output_dir=invalid_dir)


class TestCompilerErrorSurfacing(unittest.TestCase):
    """Tests that Compiler.compile() surfaces subprocess errors properly."""

    def setUp(self):
        self.work_dir = Path(tempfile.mkdtemp())
        self.config = CompilationConfig(target=CompilationTarget.TRN1)
        self.compiler = Compiler(self.config)

        # Mock IR that writes a valid .pb file when serialized
        self.mock_ir = MagicMock()
        self.mock_ir.__class__ = type("HLOModule", (), {})
        proto = MagicMock()
        proto.SerializeToString.return_value = b"fake-proto"
        self.mock_ir.to_proto.return_value = proto

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    @patch("nkipy.core.compile.isinstance", return_value=True)
    @patch("nkipy.core.compile.subprocess.run")
    def test_nonzero_exit_surfaces_stderr_and_stdout(self, mock_run, _mock_isinstance):
        """When neuronx-cc exits non-zero, the RuntimeError includes stderr and stdout."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr=b"error: invalid HLO\n",
            stdout=b"compiling graph...\n",
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.compiler.compile(
                self.mock_ir,
                self.work_dir,
                "file.neff",
            )

        msg = str(ctx.exception)
        self.assertIn("exit code 1", msg)
        self.assertIn("error: invalid HLO", msg)
        self.assertIn("compiling graph...", msg)
        self.assertIn("neuronx-cc", msg)

    @patch("nkipy.core.compile.isinstance", return_value=True)
    @patch("nkipy.core.compile.subprocess.run")
    def test_zero_exit_missing_output_surfaces_stderr(self, mock_run, _mock_isinstance):
        """When neuronx-cc exits 0 but output file is missing, stderr/stdout are shown."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr=b"warning: skipped codegen\n",
            stdout=b"",
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.compiler.compile(
                self.mock_ir,
                self.work_dir,
                "file.neff",
            )

        msg = str(ctx.exception)
        self.assertIn("expected but not generated", msg)
        self.assertIn("warning: skipped codegen", msg)

    @patch("nkipy.core.compile.isinstance", return_value=True)
    @patch("nkipy.core.compile.subprocess.run")
    def test_success_returns_output_path(self, mock_run, _mock_isinstance):
        """When compilation succeeds and output exists, return its path."""
        output_file = "file.neff"
        (self.work_dir / output_file).write_bytes(b"fake-neff")

        mock_run.return_value = MagicMock(returncode=0, stderr=b"", stdout=b"")

        result = self.compiler.compile(
            self.mock_ir,
            self.work_dir,
            output_file,
        )

        self.assertEqual(result, self.work_dir / output_file)

    @patch("nkipy.core.compile.isinstance", return_value=True)
    @patch("nkipy.core.compile.subprocess.run")
    def test_cwd_restored_after_failure(self, mock_run, _mock_isinstance):
        """Working directory is restored even when compilation fails."""
        mock_run.return_value = MagicMock(returncode=1, stderr=b"", stdout=b"")

        cwd_before = Path.cwd()
        with self.assertRaises(RuntimeError):
            self.compiler.compile(
                self.mock_ir,
                self.work_dir,
                "file.neff",
            )

        self.assertEqual(Path.cwd(), cwd_before)


if __name__ == "__main__":
    unittest.main()
