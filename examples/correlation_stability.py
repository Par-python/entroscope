"""Correlation-stability entropy example.

Run:  python examples/correlation_stability.py

Idea (from an r/dataisbeautiful thread): instead of taking the entropy of a
price series directly, take a *rolling correlation* between two assets and feed
THAT series into entroscope. Low entropy of the correlation series means the
correlation sits in a stable band; high entropy means the relationship is
wandering — a correlation-regime breakdown.

This example ships the honest comparison alongside it: the rolling standard
deviation of the same correlation series. Std-dev already measures "how much the
correlation moves". Entropy only earns its place if it separates the stable and
broken regimes at least as cleanly. The numbers below let you judge that — they
are not a claim that entropy wins.

Synthetic data lives in ``_synthetic`` so this runs with no external files. Swap
``data.correlated_assets()`` for a 2-column frame of your own returns to try it
on real data.
"""

from entroscope import permutation

try:
    from . import _synthetic as data
except ImportError:
    import _synthetic as data


def _split_means(series, half):
    """Mean of a derived series over its stable vs broken halves.

    The derived series is shorter than the raw data (rolling windows drop
    leading NaNs), so ``half`` is taken on the derived series' own length.
    """
    clean = series.dropna()
    cut = len(clean) // 2
    return clean.iloc[:cut].mean(), clean.iloc[cut:].mean()


def correlation_stability():
    """Entropy of rolling correlation vs a rolling-std baseline."""
    returns = data.correlated_assets()

    # 1. Rolling correlation between the two assets — the derived series.
    corr = returns["a"].rolling(window=30).corr(returns["b"])

    # 2. Entropy of that correlation series (ordinal, no binning needed).
    corr_entropy = permutation.rolling(corr.dropna(), window=40, order=3)

    # 3. Honest baseline: rolling std of the same correlation series.
    corr_std = corr.rolling(window=40).std()

    half = len(returns) // 2
    ent_stable, ent_broken = _split_means(corr_entropy, half)
    std_stable, std_broken = _split_means(corr_std, half)

    print("Correlation stability — two assets, regime break at the midpoint")
    print(f"  {'measure':<28}{'stable':>10}{'broken':>10}{'separation':>12}")
    print(
        f"  {'permutation entropy of corr':<28}"
        f"{ent_stable:>10.3f}{ent_broken:>10.3f}{ent_broken - ent_stable:>12.3f}"
    )
    print(
        f"  {'rolling std of corr (baseline)':<28}"
        f"{std_stable:>10.3f}{std_broken:>10.3f}{std_broken - std_stable:>12.3f}"
    )
    print(
        "  -> on THIS data the plain std baseline separates the regimes and the\n"
        "     entropy of the smooth correlation swing does not — a reminder to\n"
        "     always check an entropy signal against a trivial baseline before\n"
        "     trusting it. Entropy earns its place when the broken regime is\n"
        "     erratic/disordered, not smoothly oscillating.\n"
    )
    return corr, corr_entropy, corr_std


def main():
    print("=" * 60)
    print("CORRELATION-STABILITY ENTROPY")
    print("=" * 60)
    correlation_stability()


if __name__ == "__main__":
    main()
