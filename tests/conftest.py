"""Test isolation: point HOME at a throwaway dir *before* MetaMan is imported,
so settings, logs and the default data root never touch the real home."""

import os
import tempfile

_TMP_HOME = tempfile.mkdtemp(prefix="mm_test_home_")
os.environ["USERPROFILE"] = _TMP_HOME
os.environ["HOME"] = _TMP_HOME
