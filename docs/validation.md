# Validation

Entropy numbers are only useful if you can trust them. Every measure in
entroscope is checked against an independent implementation or a closed-form
result, and those checks run in CI on every push.

## What each measure is checked against

| Measure | Reference | Agreement |
| --- | --- | --- |
| `sample` | [antropy](https://github.com/raphaelvallat/antropy) `sample_entropy`, [EntropyHub](https://github.com/MattWillFlood/EntropyHub) `SampEn` | exact (relative error < 1e-10) |
| `approximate` | antropy `app_entropy`, EntropyHub `ApEn` | exact |
| `permutation` (+ `normalized`) | antropy `perm_entropy`, EntropyHub `PermEn` | exact |
| `spectral` (+ `normalized`) | antropy `spectral_entropy(method="fft")` | exact |
| `multiscale` | EntropyHub `MSEn` (coarse-graining, fixed tolerance) | exact |
| `shannon` | `scipy.stats.entropy` on the same histogram | exact |
| `differential` (`dist="normal"`) | `scipy.stats.norm(...).entropy()` | exact |
| `transfer` | bivariate-Gaussian closed form; Kraskov (2004) analytic mutual information | within 0.05 bits / 0.03 nats |

The comparisons run on four signal types (white noise, sine plus noise, an
AR(1) process and the chaotic logistic map) so agreement isn't an accident of
one input.

!!! note "Spectral entropy and EntropyHub"
    EntropyHub's `SpecEn` uses a different spectral estimator, so its numbers
    differ from both entroscope and antropy by design. entroscope follows
    antropy's periodogram-based definition.

## Two definitions that were corrected

Cross-checking found two places where entroscope drifted from the published
definitions. Both are fixed; results from earlier versions will differ slightly.

- **Sample entropy** now counts length-`m` and length-`m+1` template matches
  over the same `N - m` starting points (Richman & Moorman, 2000). Earlier
  versions used `N - m + 1` templates for the length-`m` count, which shifted
  values by about 0.001 to 0.003.
- **Multiscale entropy** now fixes the tolerance at `r * std` of the *original*
  series and reuses it at every scale (Costa et al., 2002). Earlier versions
  recomputed it per scale, which made white noise look *more* complex at coarser
  scales, the opposite of the textbook result.

## Run the checks yourself

```bash
pip install -e ".[dev,reference]"
pytest tests/test_reference.py -v
```

Without the `reference` extra, the live comparisons are skipped, but the pinned
values (recorded from antropy 0.2.2 and EntropyHub 2.0) still run in every test
job.
