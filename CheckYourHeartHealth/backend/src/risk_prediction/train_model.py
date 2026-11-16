"""Train and honestly evaluate the cardiovascular risk model.

Run from the backend folder:
    python src/risk_prediction/train_model.py

Data note: the repo only has a standardized copy of the Kaggle "cardio" dataset
(cardio_train_clean.csv, z-scored columns). The scaling constants below were
recovered from that file (integer grids for age/blood pressure, BMI relation
for weight), so the raw values are an approximation of the original ones.
The model is trained on those reconstructed values, so the app can feed it
normal units (years, cm, kg, mmHg).
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "cardio_train_clean.csv")
OUT_MODEL = os.path.join(HERE, "heart_risk_model.joblib")
OUT_METRICS = os.path.join(HERE, "metrics.json")

# (mean, std) used to un-standardize the columns of the repo's CSV.
UNSCALE = {
    "age": (52.8, 6.769),
    "height": (164.4, 8.185),
    "weight": (74.35, 14.33),
    "ap_hi": (126.7, 16.682),
    "ap_lo": (81.3, 9.428),
}
FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active", "bmi"]


def load_raw():
    df = pd.read_csv(DATA)
    for col, (mean, std) in UNSCALE.items():
        df[col] = df[col] * std + mean
    df["age"] = df["age"].round()
    df["ap_hi"] = df["ap_hi"].round()
    df["ap_lo"] = df["ap_lo"].round()
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    return df[FEATURES], df["cardio"].astype(int)


def make_lgb():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=42, verbose=-1)


def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "auc": round(roc_auc_score(y, p), 4),
            "f1": round(f1_score(y, pred), 4)}


def main():
    X, y = load_raw()
    print("rows:", len(X), "positive rate:", round(y.mean(), 3))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "lightgbm": make_lgb()}
    results = {"rows": int(len(X)), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)), "models": {}}
    for name, m in models.items():
        cv = cross_val_predict(m, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               method="predict_proba")[:, 1]
        m.fit(X_tr, y_tr)
        test = m.predict_proba(X_te)[:, 1]
        results["models"][name] = {"cv_on_train": scores(y_tr, cv), "held_out_test": scores(y_te, test)}
        print(name, results["models"][name])

    final = make_lgb().fit(X_tr, y_tr)
    results["deployed_model"] = "lightgbm"
    results["note"] = ("Held-out test = 20% of rows never used for fitting. Dataset is the Kaggle cardio "
                       "data (standardized copy, un-scaled approximately).")
    joblib.dump({"model": final, "features": FEATURES}, OUT_MODEL)
    with open(OUT_METRICS, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
