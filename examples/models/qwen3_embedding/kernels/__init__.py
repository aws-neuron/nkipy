"""Qwen3 Kernels"""

import os
import sys

# Ensure models/ is on sys.path for common.* imports
_models_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
