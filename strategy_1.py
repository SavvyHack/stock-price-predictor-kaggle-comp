import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

train = pd.read_csv("data/train.csv", index_col="ID")
test = pd.read_csv("data/test.csv", index_col="ID")

PRICE_COLS = [f"p{i}" for i in range(1, 41)]
K = 4000 # max stocks that can be sold


# Feature engineering
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

# Training with LightGBM
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_trn, y_val = y_train[trn_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        verbose=-1,
        random_state=42
    )

    model.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

    print(f"Fold {fold} OOF RMSE: {np.sqrt(((oof_preds[val_idx] - y_val) ** 2).mean()):.4f}")

# Computing final prediction
predicted_gain = p40_train - oof_preds
sell_model = np.zeros(len(train), dtype=int)
sell_model[np.argsort(predicted_gain)[-K:]] = 1

R = float((sell_model * (p40_train - y_train)).sum())
print(f"Estimated return: {R:.2f}")

sell_test = np.zeros(len(test), dtype=int)
sell_test[np.argsort(p40_test - test_preds)[-K:]] = 1

submission = pd.DataFrame({"ID": test.index, "sell": sell_test})
submission.to_csv("submission.csv", index=False)
print("Saved to submission.csv")