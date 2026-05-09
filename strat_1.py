import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

train = pd.read_csv("data/train.csv", index_col="ID")
test = pd.read_csv("data/test.csv", index_col="ID")

PRICE_COLS = [f"p{i}" for i in range(1, 41)]
K = 4000 # max stocks that can be sold

def get_features(df):
    p = df[PRICE_COLS].values 
    f = pd.DataFrame(index=df.index)

    # current price
    f["p"] = p[:, -1]

    # returns over different windows
    for w in [1, 5, 10, 20, 39]:
        f[f"ret_{w}"] = (p[:, -1] - p[:, -1 - w]) / (p[:, -1 - w] + 1e-9)
    
    # Volatility
    for w in [5, 10, 20]:
        f[f"vol_{w}"] = np.diff(p, axis=1)[:, -w:].std(axis=1)

    # trend slope
    for w in [5, 10, 20]:
        x = np.arange(w, dtype=float)
        xm = x - x.mean()
        Y = p[:, -w:]
        Ym = Y - Y.mean(axis=1, keepdims=True)
        f[f"slop_{w}"] = (Ym * xm).sum(axis=1) / (xm ** 2).sum()

    # mean reversion
    for w in [5, 10, 20]:
        f[f"dev_mean_{w}"] = p[:, -1] - p[:, -w:].mean(axis=1)

    # high-low range
    hi = p[:, -20:].max(axis=1)
    lo = p[:, -20:].min(axis=1)
    f["pos_in_range"] = (p[:, -1] - lo) / (hi - lo + 1e-9)

    # RSI 
    diffs = np.diff(p[:, -15:], axis=1)
    up    = np.clip(diffs, 0, None).sum(axis=1)
    down  = np.clip(-diffs, 0, None).sum(axis=1)
    f["rsi"] = up / (up + down + 1e-9)
 
    return f.fillna(0)

X_train = get_features(train)
X_test = get_features(test)
y_train = train["p50"].values
p40_train = train["p40"].values
p40_test = test["p40"].values

print(f"Features: {X_train.shape[1]}")
