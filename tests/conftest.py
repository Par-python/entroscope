"""Test-session setup: force a non-interactive matplotlib backend.

The library deliberately does NOT force a backend (see issue #2). The test
environment is responsible for declaring itself headless, which is what this
does — before matplotlib resolves an interactive backend.
"""

import os

# Set before matplotlib is imported anywhere, so it never picks a GUI backend.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
