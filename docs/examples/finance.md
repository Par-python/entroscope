# Financial time series

Measure market uncertainty over time with permutation and spectral entropy.

```python
import pandas as pd
from entropix import permutation, spectral

stock_prices = pd.read_csv("sp500.csv")["close"]
perm_entropy = permutation.rolling(stock_prices, window=50, order=3)
spec_entropy = spectral.rolling(stock_prices, window=50)
```

Permutation entropy is robust to noise and captures ordinal structure; spectral
entropy reveals whether price movement is dominated by a few cycles (low) or
broadband/noisy (high).
