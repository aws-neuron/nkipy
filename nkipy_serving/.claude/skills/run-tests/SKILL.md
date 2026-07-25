---
name: run-tests
description: Run nkipy-serving unit, integration, and AWS Neuron device tests with structured results and zero skipped tests. Use whenever the user asks to test, verify, validate, check whether changes work, run a model/device suite, or run the complete test matrix. Support cached and clean NEFF runs, recover machine-local DSV4 paths from the active agent session, and report concise failure evidence without dumping raw device logs.
---

# Run nkipy-serving Tests

Run tests from the `nkipy_serving/` package directory in the NKIPy monorepo. Resolve it from the
current workspace; do not rely on a hard-coded absolute path.

Interpret the first argument as a scope and `--clean` or `--fresh` as a request to clear compiled
NEFF caches. Read [test-catalog.md](test-catalog.md) for exact scoped commands and prerequisites.

## Scope selection

- No scope or `unit`: run unit tests only. This is intentionally the fast default.
- `integration`: run non-device integration tests.
- `device-qwen3`, `device-qwen3-moe`, `device-gpt-oss`, `device-ep`, or `device-dsv4`: run that device group.
- `device`: run all device groups sequentially.
- `all`: run the complete zero-skip matrix through `scripts/run_all_tests.sh`.

Use `../.venv/bin/python -m pytest`, not a user-level `pytest`. The workspace venv is the verified
environment and contains the editable `nkipy_serving` package.

## Pre-flight

1. Verify the environment before the first run:

   ```bash
   ../.venv/bin/python -c "import nkipy_serving; print(nkipy_serving.__file__)"
   ```

2. For `all`, `device`, or a device scope, check for an existing test/server process before killing
   anything. If stale `spawn_main` workers from a prior run own Neuron cores, terminate those stale
   workers and wait for NRT to release the cores. Never use a broad `sglang` or `nkipy` process
   pattern because it can kill the agent or unrelated work.

3. If `--clean` or `--fresh` is requested, pass `--clean` to the full runner. For a scoped device
   run, clear only that scope's cache directories listed in the catalog. State that cold compilation
   can add tens of minutes; DSV4 R1 4k can take roughly two hours to become ready on a cold cache.

4. Before a DSV4 run, recover the exact checkpoint, tokenizer, and prepared-weight paths from the
   active agent session history. Prefer paths that were previously used successfully in this same
   session. Validate each recovered path with `test -d` before starting pytest. Do not put
   machine-local paths into committed docs or skill files, and do not guess or broadly scan the
   filesystem. If the session does not contain usable paths, ask the user for them.

## Execution

Redirect verbose output to a scope-specific file under `/tmp`; device output is too large to keep
inline. Keep one log and one JUnit XML file per group so later groups do not overwrite earlier
evidence.

For the complete matrix:

```bash
PYTHON_BIN=../.venv/bin/python bash scripts/run_all_tests.sh > /tmp/nkipy_serving_all_tests.log 2>&1
```

Append `--clean` when requested. Export the three DSV4 variables recovered from session history
before invoking the script:

- `NKIPY_SERVING_DSV4_HF_MODEL_ID`
- `NKIPY_SERVING_DSV4_TOKENIZER_MODEL_ID`
- `NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR`

The full runner separates unit, non-device integration, and every device group. It writes JUnit XML
under `/tmp/nkipy_serving_pytest_xml` by default and fails a group if pytest skips any test.

For a narrower scope, use the exact command and marker filter from the catalog, redirect its output,
and add `--junitxml=/tmp/nkipy_serving_<scope>.xml`. A requested scope passes only when pytest exits
zero and the XML reports `failures=0`, `errors=0`, and `skipped=0`. A missing model or prepared-weight
path that causes a skip is a failed prerequisite, not a passing test run.

Between device groups, terminate only stale test workers and allow NRT time to release cores. If a
server fails readiness and its log contains `Logical Neuron Core(s) not available`, clean up stale
workers, verify core release, and retry that group once before reporting a code failure.

Do not stop while a required test process is still running. Poll long device runs periodically and
give the user short progress updates.

## Result extraction

Read the last pytest summary lines first. On failure, extract only the failure/error section and the
last relevant startup or compiler lines. Do not dump complete device logs.

Distinguish:

- `FAIL`: an assertion failed.
- `ERROR`: fixture, server startup, worker startup, or collection failed.
- `TIMEOUT`: the command or readiness window expired.
- `CRASH`: no pytest summary due to a process, runtime, compiler, or OOM crash.
- `SKIPPED`: a required prerequisite was absent; treat this as an incomplete/failed run.

Report a compact table with group, passed/failed/error/skipped counts, and elapsed time. Put failure
details after the table and name the corresponding `/tmp` log. For a successful complete run, state
the total number of tests and explicitly confirm zero skipped tests.
