# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for compiler wrapper functions"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from nkipy.core.compile import CompilationTarget, compile_to_neff, lower_to_nki, trace


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


if __name__ == "__main__":
    unittest.main()
