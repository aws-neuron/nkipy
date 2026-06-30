"""Reproducer: nanobind conflict between nki.compiler.kernel_builder and Spike.

Filed against: Spike / NKI runtime integration
Platform: trn2.48xlarge
NKI version: see `pip show nki`

Summary:
  When nki.compiler.kernel_builder and spike._spike are both imported in
  the same Python process (as happens in nkipy's compile+execute flow for
  the nkigen-lite backend), numerically-sensitive kernels can produce wrong
  results.  Running compilation and execution in separate processes gives
  correct results.

  For the specific values a=0.6238625646, b=0.6238614321 (where a//b=1):
    - Same-process (compile + spike execute): → 0.0 (WRONG)
    - Separate processes: → 1.0 (correct)

  The issue manifests as nanobind RuntimeWarnings at import time:
    RuntimeWarning: nanobind: type 'TensorMetadata' was already registered!
    RuntimeWarning: nanobind: type 'Spike' was already registered!

  This suggests shared native state corruption between the two libraries.

Reproduction:
  NEURON_RT_NUM_CORES=1 NEURON_RT_VISIBLE_CORES=0 python spike_floor_divide_bug.py

  This script runs two sub-processes to verify the bug is process-isolation
  dependent:
    1. Compiles the kernel and executes via nb.CompiledKernel (process A)
    2. Executes the same NEFF via nkipy/Spike runtime (process B)

  When run in separate processes (as this script does), both give correct
  results. The bug only manifests when both libraries share a process.
"""

import os
import subprocess
import sys
import tempfile
import shutil

import numpy as np


def main():
    tmpdir = tempfile.mkdtemp(prefix="spike_bug_")
    neff_path = os.path.join(tmpdir, "file.neff")
    a_path = os.path.join(tmpdir, "a.npy")
    b_path = os.path.join(tmpdir, "b.npy")

    a_val = np.float32(0.6238625646)
    b_val = np.float32(0.6238614321)
    a_np = np.full((128, 128), a_val, dtype=np.float32)
    b_np = np.full((128, 128), b_val, dtype=np.float32)

    np.save(a_path, a_np)
    np.save(b_path, b_np)

    print(f"Test: floor_divide({a_val}, {b_val})")
    print(f"  numpy a//b = {a_val // b_val:.0f} (expected: 1)")
    print()

    # Step 1: Compile and execute via kernel_builder
    script_compile = f"""
import numpy as np
import nki.compiler.kernel_builder as nb
from nkigen_lite.tensor_ir import Builder, DType
from nkigen_lite.tensor_ir.passes import lower_to_nki
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

b = Builder("fdiv")
x = b.add_input("x", (128, 128), DType.F32)
y = b.add_input("y", (128, 128), DType.F32)
z = b.floor_divide(x, y)
b.set_outputs({{"z": z}})

nki_graph = lower_to_nki(b.graph)
kernel_fn = build_kb_kernel(nki_graph)

a_np = np.load("{a_path}")
b_np = np.load("{b_path}")

# Compile
opts = nb.CompileOptions(target="trn2", output_path="{neff_path}", artifacts_dir="{tmpdir}")
z_out = np.zeros((128, 128), dtype=np.float32)
compiled = nb.compile_kernel(kernel_fn, inputs={{"x": a_np, "y": b_np}}, outputs={{"z_out": z_out}}, compile_opts=opts)

# Execute via CompiledKernel
z_result = np.zeros((128, 128), dtype=np.float32)
compiled.execute(inputs={{"x": a_np, "y": b_np}}, outputs={{"z_out": z_result}})
print(f"nb.CompiledKernel.execute: {{z_result[0,0]:.0f}}")
"""
    env = os.environ.copy()
    r1 = subprocess.run(
        [sys.executable, "-c", script_compile],
        capture_output=True, text=True, env=env, timeout=120
    )
    if r1.returncode != 0:
        print(f"Compile step FAILED:\n{r1.stderr[-500:]}")
        shutil.rmtree(tmpdir)
        return
    nb_result = r1.stdout.strip().split("\n")[-1]
    print(f"Process A (kernel_builder): {nb_result}")

    # Step 2: Execute the same NEFF via nkipy/Spike
    script_spike = f"""
import sys, os
sys.path.insert(0, os.path.join("{os.getcwd()}", "tests"))
import numpy as np
from nkipy.runtime.execute import DeviceKernel, DeviceTensor

a_np = np.load("{a_path}")
b_np = np.load("{b_path}")

dk = DeviceKernel.load_from_neff("{neff_path}", "fdiv")
print(f"NEFF inputs: {{list(dk.input_tensors_info.keys())}}")
print(f"NEFF outputs: {{list(dk.output_tensors_info.keys())}}")

device_inputs = {{
    "x": DeviceTensor.from_numpy(a_np),
    "y": DeviceTensor.from_numpy(b_np),
}}
device_outputs = {{
    "z_out": DeviceTensor.from_numpy(np.zeros((128, 128), dtype=np.float32)),
}}
dk(inputs=device_inputs, outputs=device_outputs, save_trace=False)
result = device_outputs["z_out"].numpy()
print(f"Spike DeviceKernel: {{result[0,0]:.0f}}")
"""
    r2 = subprocess.run(
        [sys.executable, "-c", script_spike],
        capture_output=True, text=True, env=env, timeout=60
    )
    if r2.returncode != 0:
        print(f"Spike step FAILED:\n{r2.stderr[-500:]}")
        shutil.rmtree(tmpdir)
        return
    spike_lines = [l for l in r2.stdout.strip().split("\n") if l.strip()]
    for line in spike_lines:
        print(f"Process B (Spike): {line}")

    # Verdict
    print()
    nb_val = nb_result.split(":")[-1].strip()
    spike_val = spike_lines[-1].split(":")[-1].strip()
    if nb_val == "1" and spike_val == "0":
        print("BUG CONFIRMED: Same NEFF, different results between runtimes.")
        print("  nb.CompiledKernel.execute() → 1 (correct)")
        print("  Spike DeviceKernel → 0 (WRONG)")
        print()
        print("The NEFF is at:", neff_path)
        print("Input a:", a_val, " Input b:", b_val)
        print("Expected result: 1 (since a > b, a//b = 1)")
        # Don't clean up so the NEFF can be inspected
        return
    elif nb_val == spike_val == "1":
        print("PASS: both runtimes agree on correct result (1)")
    else:
        print(f"UNEXPECTED: nb={nb_val}, spike={spike_val}")

    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
