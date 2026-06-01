"""Medical / biomedical entropy examples.

Run:  python examples/medical.py
Optionally saves figures to /tmp if matplotlib can write there.

Each scenario uses synthetic data from ``_synthetic`` so it runs with no
external files. Swap the generator call for ``pd.read_csv(...)['column']`` to
use your own recordings.
"""

from entroscope import sample, approximate, permutation, spectral, differential

try:  # examples run both as a script and as a module
    from . import _synthetic as data
except ImportError:
    import _synthetic as data


def _split_means(rolling, half):
    """Mean rolling entropy before vs. after the regime change at `half`."""
    before = rolling.iloc[:half].mean()
    after = rolling.iloc[half:].mean()
    return before, after


def heart_rate_hrv():
    """Heart rate / HRV — sample entropy detects reduced variability."""
    hr = data.heart_rate()
    roll = sample.rolling(hr, window=100, m=2, r=0.2)
    before, after = _split_means(roll, len(hr) // 2)
    print("Heart rate (sample entropy)")
    print(f"  healthy-phase entropy : {before:6.3f}")
    print(f"  regular-phase entropy : {after:6.3f}")
    print("  -> a drop signals reduced beat-to-beat variability\n")
    return hr, roll


def eeg_seizure():
    """EEG seizure onset — permutation & spectral entropy collapse."""
    sig = data.eeg()
    perm = permutation.rolling(sig, window=100, order=4)
    spec = spectral.rolling(sig, window=100, sf=50.0)
    half = len(sig) // 2
    pb, pa = _split_means(perm, half)
    sb, sa = _split_means(spec, half)
    print("EEG (permutation & spectral entropy)")
    print(f"  background -> seizure  permutation: {pb:5.2f} -> {pa:5.2f}")
    print(f"  background -> seizure  spectral   : {sb:5.2f} -> {sa:5.2f}")
    print("  -> both fall as the trace locks onto a rhythm\n")
    return sig, perm, spec


def respiration_irregularity():
    """Respiration — approximate entropy rises as breathing turns irregular."""
    resp = data.respiration()
    roll = approximate.rolling(resp, window=80, m=2, r=0.2)
    before, after = _split_means(roll, len(resp) // 2)
    print("Respiration (approximate entropy)")
    print(f"  steady-phase entropy   : {before:6.3f}")
    print(f"  irregular-phase entropy: {after:6.3f}")
    print("  -> a rise flags loss of a regular breathing cycle\n")
    return resp, roll


def glucose_variability():
    """Continuous glucose — differential entropy tracks variability."""
    g = data.glucose()
    roll = differential.rolling(g, window=48, dist="normal")
    before, after = _split_means(roll, len(g) // 2)
    print("Glucose (rolling differential entropy)")
    print(f"  controlled-phase entropy: {before:6.3f}")
    print(f"  volatile-phase entropy  : {after:6.3f}")
    print("  -> a rise marks a shift into a more volatile regime\n")
    return g, roll


def main():
    print("=" * 60)
    print("MEDICAL / BIOMEDICAL ENTROPY EXAMPLES")
    print("=" * 60)
    heart_rate_hrv()
    eeg_seizure()
    respiration_irregularity()
    glucose_variability()


if __name__ == "__main__":
    main()
