"""Every Python example in the README and the docs quickstart must run as written.

Blocks in one file run in order in a shared namespace, the way a reader works
through the page. `# -> value` comments that start with a number are checked
against what the line actually returns.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["README.md", "docs/quickstart.md"]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
# `expr  # -> 6.05 ...` or `expr  # -> (3.64, 0.70) ...`: the numbers to check.
CLAIM = re.compile(r"^(?P<expr>[^#]+?)\s+# -> (?P<value>\(?-?\d[\d.,\s-]*\)?)")
ASSIGNMENT = re.compile(r"^\s*[A-Za-z_][\w.\[\]]*\s*=(?!=)")


def _numbers(text):
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]


@pytest.mark.parametrize("page", PAGES)
def test_examples_run_and_claims_hold(page, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    blocks = BLOCK.findall((ROOT / page).read_text())
    assert blocks, f"no python blocks in {page}"
    namespace = {}
    for i, block in enumerate(blocks, 1):
        exec(compile(block, f"{page}:block{i}", "exec"), namespace)
        plt.close("all")
        for line in block.splitlines():
            claim = CLAIM.match(line)
            if not claim or ASSIGNMENT.match(line):
                continue
            expected = _numbers(claim["value"])
            actual = np.atleast_1d(np.asarray(eval(claim["expr"], namespace), dtype=float))
            # Values are shown rounded to 2-3 significant decimals.
            assert actual == pytest.approx(expected, abs=0.006), f"{page}: {line.strip()}"
