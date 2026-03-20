# conftest.py — Shared pytest fixtures

import sys
import os

# Ensure backend module is importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
