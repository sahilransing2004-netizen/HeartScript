import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("heart.csv")
X = df.drop(columns=["target"])
y = df["target"]

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

print("threshold  recall  precision")
for t in [0.5, 0.4, 0.3, 0.25, 0.2]:
    pred = (probs >= t).astype(int)
    print(f"{t:<10} {recall_score(y, pred):.3f}   {precision_score(y, pred):.3f}")

pipe.fit(X, y)
joblib.dump(pipe, "risk_model.joblib")
print("Saved risk_model.joblib")
