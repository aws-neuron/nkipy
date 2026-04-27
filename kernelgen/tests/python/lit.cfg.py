# python/dialects/lit.cfg.py

import os
import sys

import lit.formats

# A name for this test suite (purely cosmetic).
config.name = "NKIPy MLIR Passes Tests"

# Use the 'shell test' format: interpret RUN lines with a shell.
config.test_format = lit.formats.ShTest(execute_external=True)

# Treat .py files as tests.
config.suffixes = [".py"]

# Don't treat the config file itself as a test.
config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "__pycache__"]

# Where the test sources live.
config.test_source_root = os.path.dirname(__file__)

# Where tests are executed. Often same as source root for simple setups.
config.test_exec_root = config.test_source_root

# --- Substitutions ---

# %PYTHON -> the Python interpreter running lit (from the env you invoked `lit` in)
config.substitutions.append(("%PYTHON", sys.executable))

# Copy the current shell PYTHONPATH into lit's environment, without changing it.
# (Lit already starts from the process env, but this makes it explicit and avoids overrides.)
if "PYTHONPATH" in os.environ:
    config.environment["PYTHONPATH"] = os.environ["PYTHONPATH"]

# Use FileCheck via PATH (no hardcoded path).
filecheck_path = "FileCheck"
config.substitutions.append(("FileCheck", filecheck_path))
