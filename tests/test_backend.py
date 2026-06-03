"""Regression test for issue #2: importing entroscope must not force a backend."""

import subprocess
import sys


def test_import_does_not_change_backend():
    # Run in a subprocess with a known non-Agg backend selected via MPLBACKEND.
    # If importing entroscope calls matplotlib.use("Agg"), the backend will flip
    # to "agg" and the assertion in the child fails.
    code = (
        "import matplotlib\n"
        "before = matplotlib.get_backend().lower()\n"
        "import entroscope\n"
        "from entroscope import shannon, transfer, divergence\n"
        "after = matplotlib.get_backend().lower()\n"
        "assert before == 'template', f'unexpected start backend {before!r}'\n"
        "assert after == 'template', f'import changed backend to {after!r}'\n"
        "print('backend-stable')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={"MPLBACKEND": "template", "PATH": __import__("os").environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    )
    assert "backend-stable" in result.stdout
