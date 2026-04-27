"""
E2E test conftest.py.

Auto-applies the 'e2e' marker to all tests in this directory.
Individual tests should add 'bir_sim' or 'device' markers as appropriate.
"""

import pytest

pytestmark = [pytest.mark.e2e]
