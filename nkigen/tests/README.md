# NKIPyKernelGen Tests

## Test Structure

```
tests/
├── conftest.py                  # Root conftest: sys.path setup, --dump-ir
├── harness.py                   # Unified test harness (Mode flags, run_kernel_test)
│
├── e2e/                         # End-to-end tests (trace -> NISA -> hardware)
│   ├── conftest.py              # Auto-marks tests with 'e2e'
│   └── test_*.py
│
├── passes/                      # MLIR pass transformation tests (FileCheck)
│   ├── pass_utils.py            # Shared pass utilities
│   └── <pass_name>/             # One directory per pass
│
└── unit/                        # Python-level unit tests
```

## Running Tests

```bash
pytest tests/passes/             # all pass-level FileCheck tests
pytest tests/e2e/                # end-to-end tests (auto-skip when no Trainium device)
pytest tests/unit/               # unit tests
pytest tests/e2e/test_rope.py -v # single file
pytest -k test_add_2d            # name substring match
```

## Test modes

`Mode` flags in `harness.py` control how each test verifies its kernel:

| Mode | Meaning |
|------|---------|
| `LLVM` | LLVM JIT execution, compare to NumPy. Requires `stop_after`. |
| `HW` | Trainium hardware execution. Auto-skips when no device is detected. |
| `STRING_CHECK` | Assert compiled IR contains/excludes specific strings. |
| `FILECHECK` | Run LLVM FileCheck against the compiled IR. |

Modes are combined with `|`, e.g. `Mode.HW | Mode.STRING_CHECK`.

## Dumping IR

Pass `--dump-ir` to any test to save intermediate MLIR after every compiler
pass:

```bash
pytest tests/e2e/test_rope.py::test_rope --dump-ir -v -s
```
