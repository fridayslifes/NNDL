"""Pytest configuration: make ``src``/``app`` importable and isolate log files."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# API tests must never write into the real logs/ folder. This has to be set
# before `src.config` is imported, because LOGS_DIR is resolved at import time.
os.environ.setdefault("CHURN_LOG_DIR", tempfile.mkdtemp(prefix="churn-test-logs-"))
