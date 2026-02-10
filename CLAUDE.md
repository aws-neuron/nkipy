# CLAUDE.md

This file provides guidance to agents working on building NKIPy.

For project overview, usage, and development guide see [README.md](README.md).

## Workflow Instructions

- Use targeted tests during development (`uv run pytest tests/unit/test_file.py -k "test_fn" -v`), run the full suite (`uv run pytest tests/ -n auto`) as a final check after all changes are done.
- The repository is a **uv workspace monorepo** with two packages: **nkipy** (`nkipy/`) and **spike** (`spike/`).
