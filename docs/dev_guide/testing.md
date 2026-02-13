# Testing & Code Quality

NKIPy has three levels of testing: unit tests for NKIPy core, unit tests for Spike runtime, and integration tests.

## Running Tests

Run all tests:

```bash
uv run pytest
```

Run with verbose output:

```bash
uv run pytest -v
```

Run specific test file:

```bash
uv run pytest tests/unit/test_tensor_api.py
```

Run with coverage:

```bash
uv run pytest --cov=nkipy --cov-report=term-missing
```

## Test Categories

### NKIPy Core Tests

Unit tests for NKIPy core verify that NumPy-compatible APIs behave like NumPy and NKIPy-specific operations work correctly.

**Location:** `tests/unit/`

### Spike Runtime Tests

Unit tests for Spike verify the runtime APIs work correctly. Spike is a separate package that provides the Pythonic runtime layer for Neuron.

**Location:** `spike/tests/` or `tests/unit/`

### Integration Tests

Integration tests run through the full pipeline: tracing → HLO lowering → compilation → execution (CPU or device).

These tests verify that NKIPy kernels produce the same results as running the equivalent NumPy code on CPU.

**Location:** 
- Tests: `tests/integration/`
- Kernels: `tests/kernels/`

---

## Linting & Formatting

NKIPy uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

### Check for Linting Errors

```bash
uv run ruff check .
```

### Auto-fix Linting Errors

```bash
uv run ruff check . --fix
```

### Check Formatting

```bash
uv run ruff format --check .
```

### Auto-format Code

```bash
uv run ruff format .
```

### Ruff Configuration

Ruff is configured in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"
exclude = ["build", "third-party"]

[tool.ruff.lint]
select = ["E", "F", "I"]  # pycodestyle, pyflakes, isort

[tool.ruff.format]
quote-style = "double"
```

---

## Type Checking

NKIPy uses [mypy](https://mypy.readthedocs.io/) for static type checking.

### Run Type Checking

```bash
uv run mypy nkipy/
```

### mypy Configuration

mypy is configured in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = "neuronxcc.*"
ignore_missing_imports = true
```

---

## Pre-commit Checklist

Before committing, run:

```bash
# Run all checks
uv run ruff check .
uv run ruff format --check .
uv run mypy nkipy/
uv run pytest
```

Or as a one-liner:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy nkipy/ && uv run pytest
```

---

## Pre-commit Hooks (Automated)

As an alternative to running checks manually, you can set up pre-commit hooks to automatically run formatting and linting checks before each commit.

### Installing Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Or with uv
uv add --dev pre-commit
```

### Setting Up Pre-commit Hooks

```bash
# Install the git hook scripts
pre-commit install
```

This installs git hooks that will run automatically before each commit.

### What Gets Checked

The project's `.pre-commit-config.yaml` configures the following hooks:

- **ruff-format**: Automatically formats Python code using Ruff
- **clang-format**: Formats C++ code in the Spike runtime

### Manual Pre-commit Run

You can also run pre-commit manually on all files:

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file.py
```

### Bypassing Pre-commit Hooks

If you need to commit without running hooks (not recommended):

```bash
git commit --no-verify -m "commit message"
```

**Note:** Pre-commit hooks only cover formatting. You should still run the full test suite (`uv run pytest`) and type checking (`uv run mypy nkipy/`) before pushing changes.
