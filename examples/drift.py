"""Distribution-drift example: KL and Jensen-Shannon between two batches.

Run:  python examples/drift.py

Compares a reference batch (e.g. training data) against a drifted batch (e.g.
later production data). Higher divergence means the distributions have moved
apart. JS is symmetric and bounded to [0, 1] bits; KL is directional. Synthetic
data lives in ``_synthetic``; swap ``data.distribution_drift()`` for two batches
of your own values.
"""

from entroscope import divergence

try:
    from . import _synthetic as data
except ImportError:
    import _synthetic as data


def drift_detection():
    """Divergence between a reference batch and a drifted batch."""
    reference, drifted = data.distribution_drift()
    kl = divergence.kl(reference, drifted, bins=30)
    js = divergence.js(reference, drifted, bins=30)
    js_self = divergence.js(reference, reference.copy(), bins=30)
    print("Distribution drift (divergence, bits)")
    print(f"  KL(reference || drifted): {kl:6.3f}")
    print(f"  JS(reference,  drifted) : {js:6.3f}")
    print(f"  JS(reference,  itself)  : {js_self:6.3f}")
    print("  -> JS near 0 means no drift; larger means the batches have moved apart")
    print("     (KL and JS are relative-entropy measures between distributions)\n")
    return reference, drifted, kl, js


def main():
    print("=" * 60)
    print("DISTRIBUTION DRIFT — KL & JENSEN-SHANNON DIVERGENCE")
    print("=" * 60)
    drift_detection()


if __name__ == "__main__":
    main()
