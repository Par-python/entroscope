# Medical & biomedical signals

Four worked examples on physiological data. Each pairs a measure to the kind of
change it detects best. Runnable end-to-end in
[`examples/medical.py`](https://github.com/entropix/entropix/blob/master/examples/medical.py)
(uses synthetic data so it runs with no files — swap in `pd.read_csv(...)` for
your own recordings).

## Heart rate / HRV — sample entropy

Reduced beat-to-beat variability is clinically meaningful. Sample entropy
**drops** when a healthy, variable heart rate becomes abnormally regular.

```python
import pandas as pd
from entropix import sample

bpm = pd.read_csv("ecg.csv")["bpm"]
regularity = sample.rolling(bpm, window=100, m=2, r=0.2)
# healthy-phase mean ~2.34 -> regular-phase mean ~1.32 : a clear drop
```

## EEG seizure onset — permutation & spectral entropy

A seizure shows up as the trace collapsing onto a dominant rhythm. Both
permutation and spectral entropy **fall** as broadband background activity gives
way to a single frequency.

```python
from entropix import permutation, spectral

eeg = pd.read_csv("eeg.csv")["uv"]
perm = permutation.rolling(eeg, window=100, order=4)
spec = spectral.rolling(eeg, window=100, sf=50.0)   # sf = sampling frequency (Hz)
# background -> seizure: permutation 4.44 -> 2.68, spectral 5.03 -> 0.63
```

## Respiration regularity — approximate entropy

Approximate entropy **rises** as a steady breathing cycle breaks down into
irregular, labored breathing.

```python
from entropix import approximate

chest = pd.read_csv("respiration.csv")["expansion"]
irregularity = approximate.rolling(chest, window=80, m=2, r=0.2)
# steady-phase mean ~0.32 -> irregular-phase mean ~0.49 : a rise
```

## Continuous glucose — differential entropy

Differential entropy is a function of spread, so it **rises** when glucose
variability increases — a shift from well-controlled to a volatile regime.

```python
from entropix import differential

glucose = pd.read_csv("cgm.csv")["mgdl"]
variability = differential.rolling(glucose, window=48, dist="normal")
# controlled-phase mean ~2.98 -> volatile-phase mean ~5.29 : a clear rise
```
