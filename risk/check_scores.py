import joblib, pandas as pd

model = joblib.load("risk_model.joblib")
df = pd.read_csv("heart.csv")

tcol = "target" if "target" in df.columns else "num"
y = (df[tcol] > 0).astype(int)
X = df.drop(columns=tcol)

idx = list(y[y == 1].index[:3]) + list(y[y == 0].index[:3])
out = X.loc[idx].copy()
out["actual"] = y.loc[idx]
out["score"] = model.predict_proba(X.loc[idx])[:, 1].round(3)
print(out.to_string())
