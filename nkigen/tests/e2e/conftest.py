"""
E2E test conftest.py.

Auto-applies the 'e2e' marker to all tests in this directory.
"""

import pytest

pytestmark = [pytest.mark.e2e]
