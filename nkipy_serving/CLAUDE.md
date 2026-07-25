# nkipy-serving

- `README.md` — project overview, architecture, how to run/test
- `../CLAUDE.md` — NKIPy monorepo context

## Engineering Constraints

- Keep PyTorch and JAX out of the runtime and model paths.
- Fail fast instead of adding silent runtime fallbacks or compatibility shims.
- The scheduler must not initialize NRT or claim NeuronCores; device work belongs
  in workers.
- Keep Python package code under `src/nkipy_serving/`.
- Keep machine-local checkpoint paths and scratch notes out of committed docs.

## Skills

- `/run-tests [scope] [--clean]` — run unit/device tests and report structured results (summary table + failure tracebacks only, never raw logs)
