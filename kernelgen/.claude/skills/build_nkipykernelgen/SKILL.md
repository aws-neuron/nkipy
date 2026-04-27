---
name: build_nkipykernelgen
description: Rebuild NKIPyKernelGen (C++ passes and Python package)
user-invocable: true
---

## Usage

`/build_nkipykernelgen`

## Instructions

Run the build script. Use `bash` (not `sh`) since it uses `source`. Use a timeout of 300000ms.

```bash
bash .claude/skills/build_nkipykernelgen/scripts/build.sh
```

Note: Run this from the NKIPyKernelGen repo root.

## Important

`pip install -e .` builds BOTH the C++ passes (nkipy-opt binary) AND the Python package in one step. There is NO need to run cmake separately — the pyproject.toml build system handles the full C++ compilation via cmake internally.
