"""Train the stacking ensemble and compare it with each single model on the same held-out test set.

Run from the backend folder:  python src/risk_prediction/train_ensemble.py
Training is resumable: finished folds are cached in .ens_cache/ so an interrupted run picks up where it stopped.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import load_raw, make_lgb  # noqa: E402
import ensemble as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".ens_cache")
BUDGET = float(os.environ.get("ENS_BUDGET_SECONDS", "1e9"))
N_FOLDS = 5
os.makedirs(CACHE, exist_ok=True)
T0 = time.time()


def make_rf():
    return RandomForestClassifier(n_estimators=250, min_samples_leaf=30, max_depth=10, n_jobs=-1, random_state=42)


def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4), "auc": round(roc_auc_score(y, p), 4),
            "f1": round(f1_score(y, pred), 4)}


def cached(name, fn):
    path = os.path.join(CACHE, name + ".npy")
    if os.path.exists(path):
        return np.load(path)
    if time.time() - T0 > BUDGET:
        print("time budget used, run again to continue"); sys.exit(3)
    out = fn(); np.save(path, out); print("done", name, round(time.time() - T0), "s", flush=True)
    return out


def main():
    X, y = load_raw()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True); y_tr = y_tr.values; y_te = y_te.values
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(X_tr, y_tr))

    oof = np.zeros((len(X_tr), 3))
    for k, (a, b) in enumerate(folds):
        oof[b, 0] = cached(f"lgb_f{k}", lambda: make_lgb().fit(X_tr.iloc[a], y_tr[a]).predict_proba(X_tr.iloc[b])[:, 1])
        oof[b, 1] = cached(f"rf_f{k}", lambda: make_rf().fit(X_tr.iloc[a], y_tr[a]).predict_proba(X_tr.iloc[b])[:, 1])
        oof[b, 2] = cached(f"tf_f{k}", lambda: E.predict_transformer(E.fit_transformer(Xs_tr[a], y_tr[a]), Xs_tr[b]))

    # Final base models on the whole training set, scored on the untouched test set
    lgb_m = make_lgb().fit(X_tr, y_tr); rf_m = make_rf().fit(X_tr, y_tr)
    tf_path = os.path.join(CACHE, "tf_final.pt")
    if os.path.exists(tf_path):
        tf_m = E.TabTransformer(); tf_m.load_state_dict(__import__("torch").load(tf_path))
    else:
        if time.time() - T0 > BUDGET:
            print("time budget used, run again to continue"); sys.exit(3)
        tf_m = E.fit_transformer(Xs_tr, y_tr); __import__("torch").save(tf_m.state_dict(), tf_path)
    test_base = np.c_[lgb_m.predict_proba(X_te)[:, 1], rf_m.predict_proba(X_te)[:, 1], E.predict_transformer(tf_m, Xs_te)]

    meta = LogisticRegression(max_iter=1000).fit(oof, y_tr)
    stack_test = meta.predict_proba(test_base)[:, 1]
    res = {"test_rows": int(len(X_te)), "train_rows": int(len(X_tr)),
           "held_out_test": {"lightgbm": scores(y_te, test_base[:, 0]), "random_forest": scores(y_te, test_base[:, 1]),
                             "transformer": scores(y_te, test_base[:, 2]), "stacked_ensemble": scores(y_te, stack_test)},
           "out_of_fold_on_train": {"lightgbm": scores(y_tr, oof[:, 0]), "random_forest": scores(y_tr, oof[:, 1]),
                                    "transformer": scores(y_tr, oof[:, 2]), "stacked_ensemble": scores(y_tr, meta.predict_proba(oof)[:, 1])},
           "meta_weights": dict(zip(["lightgbm", "random_forest", "transformer"], meta.coef_[0].round(3).tolist()))}
    best = max(res["held_out_test"], key=lambda k: res["held_out_test"][k]["auc"])
    res["best_on_held_out_auc"] = best
    json.dump(res, open(os.path.join(HERE, "ensemble_metrics.json"), "w"), indent=2)
    joblib.dump({"scaler": scaler, "lgb": lgb_m, "rf": rf_m, "meta": meta, "features": E.FEATURES,
                 "transformer_state": tf_m.state_dict()}, os.path.join(HERE, "heart_risk_ensemble.joblib"), compress=3)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
