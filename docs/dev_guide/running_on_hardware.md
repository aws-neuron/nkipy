# Running on Neuron Hardware

Practical runtime tricks for running tests, benchmarks, and examples on a
multi-core Neuron instance (e.g. `trn2.48xlarge`, which exposes 16 NeuronCores
across 4 devices). These are mostly about **NeuronCore selection** — picking
which cores a process runs on so that concurrent jobs don't collide.

## Inspecting the device

`neuron-ls` shows every Neuron device, its cores, and which PIDs currently hold
them:

```bash
neuron-ls
```

The `NEURON CORE IDS` and `PID` columns tell you which cores are busy. A core
with a `PID` is in use; an idle core is free to claim. Use this before starting
a job to find free cores and avoid contending with someone else's run.

## Selecting specific cores

`NEURON_RT_VISIBLE_CORES` restricts a process to a subset of cores. It accepts a
single index or an inclusive range:

```bash
# Run on core 0 only
NEURON_RT_VISIBLE_CORES=0 python my_script.py

# Run on cores 4,5,6,7 (e.g. a TP=4 job on the second device)
NEURON_RT_VISIBLE_CORES=4-7 torchrun --nproc-per-node 4 my_script.py
```

This is the key to running two jobs at once without them fighting over cores —
for example benchmarking two branches side by side: pin one to `0-3` and the
other to `4-7`. They land on different physical devices and run concurrently.

`NEURON_RT_NUM_CORES=N` is a related knob that just requests `N` cores without
pinning specific indices; prefer `NEURON_RT_VISIBLE_CORES` when you need
determinism about *which* cores are used.

## How the test suite isolates cores

`tests/conftest.py` already handles core isolation for `pytest-xdist` parallel
runs, so you normally don't set anything by hand:

- `pytest_xdist_auto_num_workers` caps `-n auto` at the number of visible cores
  (`Spike.get_visible_neuron_core_count()`), since each worker needs its own
  core. Asking for more workers than cores causes allocation failures.
- `pytest_configure` maps each xdist worker (`gw0`, `gw1`, …) to one core by
  setting `NEURON_RT_VISIBLE_CORES=<worker index>` and `NEURON_RT_NUM_CORES=1`.

Implications when running tests:

```bash
# Let xdist auto-size to the available cores (recommended)
uv run pytest tests/ -n auto

# To leave some cores free for another job, run on a subset and cap workers
NEURON_RT_VISIBLE_CORES=0-3 uv run pytest tests/ -n 4
```

With `-n auto`, if some cores are already busy you should still constrain
`NEURON_RT_VISIBLE_CORES` to the free ones, otherwise a worker may try to claim
an occupied core.

## Benchmarking a specific branch (git worktrees)

When comparing branches, a git worktree gives each branch its own checkout. Two
gotchas:

1. **Make sure you import the code under test.** An editable install
   (`nkigen_lite`, etc.) may resolve to the *main* checkout, not the worktree.
   Put the worktree's source first on `PYTHONPATH` so the process picks it up:

   ```bash
   PYTHONPATH=/path/to/worktree/nkigen-lite/src:$PYTHONPATH python -c \
     "import nkigen_lite, os; print(os.path.dirname(nkigen_lite.__file__))"
   ```

   Verify the printed path points into the worktree before trusting the numbers.

2. **Pin the two runs to different cores** so they can run at the same time
   under the same host/thermal conditions (a fairer comparison than
   back-to-back runs):

   ```bash
   # branch A, main checkout
   NEURON_RT_VISIBLE_CORES=0-3 torchrun --nproc-per-node 4 evaluate.py --benchmark ...

   # branch B, worktree (different cores + worktree on PYTHONPATH)
   NEURON_RT_VISIBLE_CORES=4-7 \
   PYTHONPATH=/path/to/worktree/nkigen-lite/src:$PYTHONPATH \
     torchrun --nproc-per-node 4 evaluate.py --benchmark ...
   ```

You can inspect the environment of a running process to confirm what it actually
picked up:

```bash
cat /proc/<pid>/environ | tr '\0' '\n' | grep -E 'NEURON_RT_VISIBLE_CORES|PYTHONPATH'
```

## Deprecated / ignored runtime env vars

Neuron Runtime evolves; some env vars are silently ignored on newer runtimes.
Watch the startup logs for `WARN NRT:nrt_config_parse_init_config` lines. For
example, `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS` is no longer supported on
Neuron Runtime 2.0 ("implicit async exec mode has been removed") — setting it
has no effect, and async now requires the explicit Neuron Runtime async APIs. If
a perf knob doesn't move the numbers, check whether the runtime logged that it
ignored it.
