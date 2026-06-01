# Food trend analysis (NextOnMenu)

Detect when a food ingredient's search interest stops being random — entropy
drops as a trend consolidates before going mainstream.

```python
import pandas as pd
from entropix import shannon

matcha_trends = pd.read_csv("matcha_trends.csv")["interest"]
entropy = shannon.rolling(matcha_trends, window=20)
fig = shannon.plot(matcha_trends, window=20, title="Matcha — entropy over time")
```

A sustained drop in rolling Shannon entropy means search interest is becoming
concentrated/structured rather than noisy — an early signal of a trend.
