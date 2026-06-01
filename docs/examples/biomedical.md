# Biomedical signals

Detect regularity in physiological signals such as heart rate.

```python
import pandas as pd
from entropix import sample

heart_rate = pd.read_csv("ecg.csv")["bpm"]
regularity = sample.rolling(heart_rate, window=100, m=2, r=0.2)
```

Lower sample entropy means a more regular, predictable signal; rising sample
entropy can flag a transition to a less regular regime.
