"""
Unit test conftest.py.

Auto-applies the 'unit' marker to all tests in this directory.
"""

import pytest

pytestmark = [pytest.mark.unit]
