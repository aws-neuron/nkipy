"""
APIs for use inside traced functions.

This module contains APIs that are meant to be used within functions
decorated with @trace, such as control flow constructs and optimization hints.
"""

from .knob import knob
from .control_flow import fori_loop

__all__ = ["knob", "fori_loop"]
