"""Cross-check entroscope against independent implementations.

Two layers:

- **Pinned** values produced by antropy 0.2.2, EntropyHub 2.0 and scipy on a
  fixed seed. Always run, so every CI job catches numeric drift.
- **Live** comparisons that call antropy and EntropyHub directly on several
  signal types. Skipped unless they are installed
  (``pip install "entroscope[reference]"``); a dedicated CI job installs them.
"""

import numpy as np
import pytest
import scipy.stats as st

from entroscope import approximate, differential, multiscale, permutation, sample, shannon, spectral

REL = 1e-10
WHITE = np.random.RandomState(0).randn(500)


def _signals(n=1000):
    rng = np.random.RandomState(0)
    t = np.arange(n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + rng.randn()
    logistic = np.empty(n)
    logistic[0] = 0.4
    for i in range(1, n):
        logistic[i] = 3.9 * logistic[i - 1] * (1 - logistic[i - 1])
    return {
        "white": rng.randn(n),
        "sine+noise": np.sin(2 * np.pi * t / 50) + 0.3 * rng.randn(n),
        "ar1": ar,
        "logistic": logistic,
    }


SIGNALS = _signals()


@pytest.fixture(params=sorted(SIGNALS))
def signal(request):
    return SIGNALS[request.param]


# ---------------------------------------------------------------------------
# Pinned (no optional dependencies)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, expected",
    [
        (sample.compute, 2.223148262346011),  # antropy.sample_entropy == EntropyHub.SampEn
        (approximate.compute, 1.3386285487754925),  # antropy.app_entropy
        (permutation.compute, 2.577059057137056),  # antropy.perm_entropy
        (spectral.compute, 7.29267833570191),  # antropy.spectral_entropy(method="fft")
    ],
    ids=["sample", "approximate", "permutation", "spectral"],
)
def test_pinned_reference_values(fn, expected):
    assert fn(WHITE) == pytest.approx(expected, rel=REL)


def test_pinned_multiscale_matches_entropyhub():
    expected = [2.223148262346011, 1.907161146385372, 1.7565135301663801, 1.505482878385009]
    assert list(multiscale.compute(WHITE, scales=range(1, 5)).values()) == pytest.approx(
        expected, rel=REL
    )


def test_shannon_matches_scipy(signal):
    counts, _ = np.histogram(signal, bins=10)
    assert shannon.compute(signal) == pytest.approx(st.entropy(counts, base=2), rel=REL)


def test_differential_normal_matches_scipy(signal):
    expected = st.norm(scale=np.std(signal)).entropy()
    assert differential.compute(signal) == pytest.approx(expected, rel=REL)


# ---------------------------------------------------------------------------
# Live (optional: antropy, EntropyHub)
# ---------------------------------------------------------------------------


def test_live_antropy(signal):
    ant = pytest.importorskip("antropy")
    assert sample.compute(signal) == pytest.approx(ant.sample_entropy(signal, 2), rel=REL)
    assert approximate.compute(signal) == pytest.approx(ant.app_entropy(signal, 2), rel=REL)
    assert permutation.compute(signal) == pytest.approx(ant.perm_entropy(signal, 3), rel=REL)
    assert permutation.normalized(signal) == pytest.approx(
        ant.perm_entropy(signal, 3, normalize=True), rel=REL
    )
    assert spectral.compute(signal) == pytest.approx(
        ant.spectral_entropy(signal, sf=1, method="fft"), rel=REL
    )
    assert spectral.normalized(signal) == pytest.approx(
        ant.spectral_entropy(signal, sf=1, method="fft", normalize=True), rel=REL
    )


def test_live_entropyhub(signal):
    eh = pytest.importorskip("EntropyHub")
    r = 0.2 * np.std(signal)
    assert sample.compute(signal) == pytest.approx(eh.SampEn(signal, m=2, r=r)[0][2], rel=REL)
    assert approximate.compute(signal) == pytest.approx(eh.ApEn(signal, m=2, r=r)[0][2], rel=REL)
    assert permutation.compute(signal) == pytest.approx(
        eh.PermEn(signal, m=3, Logx=2)[0][-1], rel=REL
    )
    mse = eh.MSEn(signal, eh.MSobject("SampEn", m=2, r=r), Scales=5)[0]
    ours = list(multiscale.compute(signal, scales=range(1, 6)).values())
    assert ours == pytest.approx(list(mse), rel=REL)
