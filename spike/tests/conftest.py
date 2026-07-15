def pytest_addoption(parser):
    parser.addoption(
        "--test-mode",
        default="correctness",
        choices=["correctness", "overlapping"],
        help=(
            "Test mode: 'correctness' uses small random tensors to verify numerical accuracy; "
            "'overlapping' uses large zero tensors to demonstrate async operation overlap."
        ),
    )
    parser.addoption(
        "--num-pipelines",
        type=int,
        default=3,
        help="Number of concurrent pipelines to allocate and run in test_spike_async.py tests.",
    )
    parser.addoption(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of untimed warmup iterations before measurement in the perf test.",
    )
    parser.addoption(
        "--measure-iters",
        type=int,
        default=10,
        help="Number of timed measurement iterations in the perf test.",
    )
