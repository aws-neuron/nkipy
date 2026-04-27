# NKIPyKernelGen Tests

## Test Structure

```
tests/
├── conftest.py                  # Root conftest: sys.path setup, --test-mode, --device
├── pytest.ini                   # (at project root)
│
├── test_basic_ops.py            # Element-wise ops (add, sub, mul, div) — LLVM JIT
├── test_broadcast_ops.py        # Broadcasting ops — LLVM JIT
├── test_elementwise_ops.py      # More elementwise ops — LLVM JIT
├── test_execution_engine.py     # Execution engine basics — LLVM JIT
├── test_for_loops.py            # Loop constructs — LLVM JIT
├── test_import_compatibility.py # Import smoke test
├── test_matrix_ops.py           # Matmul, combined ops — LLVM JIT
├── test_unary_ops.py            # Unary ops (exp, sin, sqrt) — LLVM JIT
├── test_reduction_ops.py        # Reductions (sum, mean, max) — LLVM JIT + BIR sim
├── test_partial_tiling.py       # Partial tiling — LLVM JIT + BIR sim
├── test_full_pipeline.py        # Full NISA pipeline — BIR sim
├── test_qwen3_kernels.py        # Qwen3 model kernels — LLVM JIT + BIR sim
│
├── e2e/                         # End-to-end pipeline tests (BIR simulation)
│   ├── conftest.py              # Auto-marks tests with 'e2e'
│   ├── test_feedforward.py
│   ├── test_matmul_add.py
│   └── test_sigmoid.py
│
├── passes/                      # MLIR pass transformation tests
│   ├── conftest.py              # Auto-marks tests with 'passes', shared fixtures
│   ├── pass_utils.py            # Shared pass utilities (FileCheck, compilation)
│   ├── annotate_memory_space/   # Each pass has: test_*.py, utils.py, outputs/
│   ├── canonicalize_loop_step/
│   ├── cleanup_bufferization_artifacts/
│   ├── eliminate_same_memspace_copy/
│   ├── eliminate_uninitialized_copies/
│   ├── knob_driven_tiling/
│   └── legalize_layout/
│
└── python/                      # LIT FileCheck tests (run via `lit`, NOT pytest)
    ├── lit.cfg.py
    ├── passes/
    └── rewrites/
```

## Running Tests

All commands run from the project root (`NKIPyKernelGen/`).

### Test modes

The `--test-mode` flag controls which tests run:

```bash
# CPU mode (default) — runs LLVM JIT + BIR simulation tests, no hardware needed
pytest
pytest --test-mode=cpu

# Device mode — runs only tests marked 'device', targeting a specific Trainium generation
pytest --test-mode=device --device=trn1
pytest --test-mode=device --device=trn2
pytest --test-mode=device --device=trn3

# All mode — CPU tests + device tests (if --device is given)
pytest --test-mode=all --device=trn2
```

### Filter by marker

```bash
pytest -m bir_sim          # only BIR simulation tests
pytest -m e2e              # only end-to-end tests
pytest -m passes           # only MLIR pass tests
pytest -m "not bir_sim"    # skip BIR simulation tests
```

### Filter by path or name

```bash
pytest tests/test_basic_ops.py -v                    # single file
pytest tests/passes/knob_driven_tiling/ -v            # single pass directory
pytest tests/test_basic_ops.py::TestElementWiseOps -v # single class
pytest -k test_add_2d                                 # name substring match
```

### LIT FileCheck tests

LIT tests are excluded from pytest. Run them separately:

```bash
lit tests/python/ -v
```

### Available markers

| Marker   | Meaning |
|----------|---------|
| `llvm_sim` | LLVM JIT simulation test (CPU) |
| `bir_sim`  | BIR simulation test (CPU) |
| `device`   | Requires Trainium hardware |
| `trn1` / `trn2` / `trn3` | Targets a specific Trainium generation |
| `e2e`      | End-to-end pipeline test |
| `passes`   | MLIR pass transformation test |

## Test Design

Each test:
1. Defines a function using NumPy operations
2. Traces the function with `@trace` decorator to generate MLIR
3. Uses `verify_against_numpy` to compare MLIR/LLVM execution with NumPy CPU execution
4. Asserts that results match within tolerance

```python
def test_new_operation(self):
    def my_func(A, B):
        return np.some_operation(A, B)

    traced_func = trace(input_specs=[((4, 3), "f32"), ((4, 3), "f32")])(my_func)

    A = np.random.randn(4, 3).astype(np.float32)
    B = np.random.randn(4, 3).astype(np.float32)

    matches, mlir_result, numpy_result = verify_against_numpy(
        traced_func, my_func, [A, B]
    )
    assert matches, "MLIR result does not match NumPy result"
```

## Import setup

`tests/conftest.py` adds `tests/passes/` to `sys.path` at pytest startup. This means:
- Pass tests can do `from pass_utils import ...`
- All tests can do `from harness import ...` and `from nkipy_kernelgen import ...` (installed package)

Each per-pass `conftest.py` adds its own directory to `sys.path` so `from utils import ...` resolves to the local `utils.py`.

## Known issues

Some test files import `convert_linalg_to_nisa` which may not be available depending on the build. These tests will show collection errors but are unrelated to the test infrastructure — the underlying transform is conditionally compiled.
