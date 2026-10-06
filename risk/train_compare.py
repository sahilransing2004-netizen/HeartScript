import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

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

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             eval_metric="logloss", random_state=42),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

rows = []
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    row = {"model": name}
    for m in scoring:
        row[m] = round(res["test_" + m].mean(), 3)
        row[m + "_std"] = round(res["test_" + m].std(), 3)
    rows.append(row)
    print(name, {m: row[m] for m in scoring})

pd.DataFrame(rows).to_csv("results.csv", index=False)
print("Saved results.csv")
