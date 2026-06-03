"""Transfer-entropy example: directional information flow between two assets.

Run:  python examples/information_flow.py

Transfer entropy is directional: TE(A->B) measures how much A's past helps
predict B's future beyond B's own past. Here A leads B by construction, so
TE(A->B) should clearly exceed TE(B->A). Synthetic data lives in ``_synthetic``;
swap ``data.lead_lag_assets()`` for a 2-column frame of your own returns.
"""

from entroscope import transfer

try:
    from . import _synthetic as data
except ImportError:
    import _synthetic as data


def information_flow():
    """Directional transfer entropy between a leading and a lagging asset."""
    df = data.lead_lag_assets()
    te_ab = transfer.compute(df["a"], df["b"], method="ksg", k=4)
    te_ba = transfer.compute(df["b"], df["a"], method="ksg", k=4)
    print("Information flow (KSG transfer entropy, bits)")
    print(f"  TE(A -> B): {te_ab:6.3f}")
    print(f"  TE(B -> A): {te_ba:6.3f}")
    print(f"  net A->B  : {te_ab - te_ba:6.3f}")
    print("  -> A leads B by construction, so TE(A->B) should dominate\n")
    return df, te_ab, te_ba


def main():
    print("=" * 60)
    print("TRANSFER ENTROPY — DIRECTIONAL INFORMATION FLOW")
    print("=" * 60)
    information_flow()


if __name__ == "__main__":
    main()
