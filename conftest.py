"""
Root conftest.py — pytest collection-time guards.

Purpose
-------
1. Ensure the project root is on sys.path regardless of invocation style
   (bare `pytest` vs `python -m pytest`). The pyproject.toml `pythonpath = ["."]`
   setting covers pytest >=7; this guard covers older versions and edge cases.

2. Pre-import torch_geometric *once* during conftest setup so that the
   Windows DLL load (torch_cluster, torch_scatter, torch_sparse) happens in a
   single controlled call rather than racing N times across parallel test-file
   collection. Errors are suppressed here — individual tests that need torch
   will fail with a clear ImportError rather than a silent process death.

Background
----------
On Windows, torch_cluster/_version_cpu.pyd and torch_scatter/_version_cpu.pyd
can raise a fatal OS exception (0xc0000139 STATUS_ENTRYPOINT_NOT_FOUND) that
bypasses Python's exception handler and kills the pytest process during
collection. torch_geometric wraps these with try/except + UserWarning, but
the crash can occur before that handler runs if multiple test files trigger the
import simultaneously. Pre-loading here forces the DLL resolution to happen
once, with torch_geometric's own handler in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Belt-and-suspenders: ensure project root on sys.path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Pre-load torch_geometric to settle DLL resolution before test collection.
# If it fails, tests that import mitre_core will report a clear ImportError.
try:
    import torch_geometric  # noqa: F401
except Exception:
    pass
