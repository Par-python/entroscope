"""Business / operational entropy examples.

Run:  python examples/business.py

Each scenario uses synthetic data from ``_synthetic`` so it runs with no
external files. Swap the generator call for ``pd.read_csv(...)['column']`` to
use your own data (sales exports, server logs, price feeds, sensor dumps).
"""

from entropix import shannon, spectral, permutation, multiscale

try:
    from . import _synthetic as data
except ImportError:
    import _synthetic as data


def _split_means(rolling, half):
    return rolling.iloc[:half].mean(), rolling.iloc[half:].mean()


def sales_demand():
    """Daily sales — rolling Shannon entropy falls as demand stabilizes."""
    s = data.daily_sales()
    roll = shannon.rolling(s, window=21, bins=10)
    before, after = _split_means(roll, len(s) // 2)
    print("Daily sales (rolling Shannon entropy)")
    print(f"  erratic-launch entropy : {before:6.3f}")
    print(f"  settled-pattern entropy: {after:6.3f}")
    print("  -> a drop means demand has consolidated into a stable shape\n")
    return s, roll


def web_traffic_anomaly():
    """Web traffic — spectral entropy jumps when bot bursts break the cycle."""
    t = data.web_traffic()
    roll = spectral.rolling(t, window=72, sf=24.0)
    before, after = _split_means(roll, len(t) // 2)
    print("Web traffic (spectral entropy)")
    print(f"  clean-cycle entropy : {before:6.3f}")
    print(f"  bot-anomaly entropy : {after:6.3f}")
    print("  -> a rise means power spread across many frequencies (anomaly)\n")
    return t, roll


def price_volatility():
    """Stock prices — permutation entropy rises with market turbulence."""
    p = data.stock_prices()
    roll = permutation.rolling(p, window=50, order=3)
    before, after = _split_means(roll, len(p) // 2)
    print("Stock prices (permutation entropy)")
    print(f"  calm-market entropy     : {before:6.3f}")
    print(f"  turbulent-market entropy: {after:6.3f}")
    print("  -> a rise gauges increasing disorder in price moves\n")
    return p, roll


def quality_control():
    """QC sensor — multiscale entropy profiles in- vs out-of-control halves."""
    sig = data.qc_sensor()
    half = len(sig) // 2
    in_control = multiscale.compute(sig.iloc[:half], scales=range(1, 6))
    out_control = multiscale.compute(sig.iloc[half:], scales=range(1, 6))
    print("QC sensor (multiscale sample entropy by scale)")
    print(f"  {'scale':>5}{'in-control':>14}{'out-of-control':>16}")
    for scale in sorted(set(in_control) | set(out_control)):
        a = in_control.get(scale, float("nan"))
        b = out_control.get(scale, float("nan"))
        print(f"  {scale:>5}{a:>14.3f}{b:>16.3f}")
    print("  -> diverging profiles flag a process leaving its controlled state\n")
    return sig, in_control, out_control


def main():
    print("=" * 60)
    print("BUSINESS / OPERATIONAL ENTROPY EXAMPLES")
    print("=" * 60)
    sales_demand()
    web_traffic_anomaly()
    price_volatility()
    quality_control()


if __name__ == "__main__":
    main()
