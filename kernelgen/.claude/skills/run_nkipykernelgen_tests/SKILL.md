---
name: run_nkipykernelgen_tests
description: Run NKIPyKernelGen tests (without rebuilding)
user-invocable: true
---

## Usage

`/run_nkipykernelgen_tests [scope]`

Where `scope` is: `all` (default), `passes`, `e2e`, or a specific path like `passes/infer_layout` or `e2e/nkipy_tests`.

## Instructions

1. Run the script at `~/.claude/skills/run_nkipykernelgen_tests/scripts/run_tests.sh` with the requested scope as the argument. Use `bash` to invoke it (not `sh`) since it uses `source`. Use a timeout of 600000ms.

```bash
bash .claude/skills/run_nkipykernelgen_tests/scripts/run_tests.sh <scope>
```

Note: Run this from the NKIPyKernelGen repo root.

2. The script saves full test output to `/tmp/nkipykernelgen_test_results.txt`. After the script finishes, use the Read tool to read that file for the complete results. This avoids context window issues with long test output.

3. When reporting results, summarize:
   - Total passed/failed/xfailed/xpassed/skipped counts
   - List any unexpected failures (FAILED, not XFAIL)
   - Note any XPASS (unexpected passes) that indicate xfail markers should be removed
