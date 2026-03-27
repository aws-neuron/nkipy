#!/usr/bin/env python3
"""Wrapper that launches torchrun under scalene.

Called by ``test.sh --profile``, not directly. Scalene initializes in this
process and ``--profile-all`` causes ``redirect_python`` to replace
``python`` on PATH. When torchrun spawns workers via subprocess, they pick
up the redirected python and each gets its own dormant scalene instance
(``--off``). Only rank 0's KernelProfiler calls
``scalene_profiler.start()``/``stop()`` to activate CPU profiling.
"""

import sys
import torch.distributed.run as torchrun_main

if __name__ == "__main__":
    sys.exit(torchrun_main.main())
