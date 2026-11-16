# CheckYourHeartHealth

A Streamlit app that estimates heart disease risk from basic health data. I built it as a student project, in a rush, and the first version made some things look like they worked when they didn't. This README is the honest version after going back through the code.

It is a learning project. Don't use it to make real health decisions.

## What actually works

- **Risk prediction.** You enter age, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol and activity. A LightGBM model trained on the Kaggle cardiovascular disease dataset (about 65,500 rows after cleaning) gives a risk percentage.
- **Accounts and roles.** Patient, doctor and admin logins, backed by SQLite. Doctors can see linked patients, leave notes and get alerts.
- **History and what-if.** Repeat risk scores are saved per patient. What-if scenarios run through the risk model. The factor breakdown still uses hand-written rules, not SHAP or LIME.
- **3D heart viewer.** An open-source Three.js heart model embedded in a page.

## Real numbers

Measured on a held-out 20% test split (13,110 rows the model never saw during training). `train_model.py` and `train_ensemble.py` in `src/risk_prediction/` reproduce them.

| Model | Accuracy | AUC |
|---|---|---|
| Logistic regression (baseline) | 72.3% | 0.789 |
| Random forest | 73.1% | 0.800 |
| Transformer (small, tabular) | 73.4% | 0.799 |
| LightGBM (what the live app uses) | 73.4% | 0.801 |
| Stacked ensemble (LightGBM + RF + Transformer, logistic-regression combiner) | 73.5% | 0.801 |

The ensemble is within noise of LightGBM. The measured gap is just 0.07 percentage points of accuracy. I have not measured confidence intervals or repeated-seed variation, so I cannot call that a meaningful improvement. These runs do not prove a hard limit on the dataset. I built the ensemble to learn stacking.

5-fold cross-validation on the training part gave 73.1% accuracy and 0.799 AUC for LightGBM.

The trained ensemble file (about 5 MB) is not in the repo, because I could not upload it from the GitHub web page. Run `python src/risk_prediction/train_ensemble.py` (about 7 minutes) and the app picks it up automatically. Without it the app uses LightGBM.

One caveat: the repo only has a standardized copy of the dataset, so I recovered the original units (years, cm, kg, mmHg) from it. They are close, not guaranteed identical to the original file.

## What does not work (yet)

- **ECG photos.** Not supported. The old hardcoded "Normal Beat, 95%" result is gone. The new classifier accepts signals only, as described below.
- **The old stacking ensemble (Transformer + LightGBM + Random Forest + MLP).** The app never used it, and it was trained on standardized data while the app sent raw values. The old "~70% accuracy, 0.79 AUC" came from a 2,622-row test set. I replaced it with the retrained ensemble in the table above.
- **Genetic algorithm / PSO feature selection.** Experiments only. The PSO run in `logs/` failed. The app doesn't use either.
- **SHAP and LIME.** Not wired to the real model. The factor explanations in the app are rule-based estimates.

## ECG signals

I trained a small residual 1D-CNN on raw [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) signals, not images. It predicts five overlapping diagnostic superclasses: NORM, MI, STTC, CD and HYP. This is separate from the tabular risk model.

| ECG model | Held-out macro AUC |
|---|---|
| LightGBM on hand-made signal features | 0.8764 |
| 1D-CNN (the ECG app model) | 0.9303 |

CNN per-class AUC: NORM 0.9509, MI 0.9295, STTC 0.9386, CD 0.9236, HYP 0.9089. These are single-run results, not accuracy. No confidence intervals or clinical calibration checks yet.

The official patient-wise folds are 1-8 for training (17,084 records), 9 for validation and best-epoch selection (2,146), 10 for test (2,158). No patient appears in more than one split. I excluded records without a diagnostic superclass. The CNN trained for 12 epochs on CPU. LightGBM was a fixed feature baseline, not an exhaustive search.

Upload a CSV with 1000 rows (10 seconds at 100 Hz), millivolt values, and columns `I,II,III,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6`. No timestamp/index column. Scores can overlap and do not add to 100%. They are not a diagnosis or a safe way to rule out disease. Do not upload real patient data to a public demo. Training scripts and data setup are in `CheckYourHeartHealth/backend/ecg_training/`.

Data credit: Wagner et al., PTB-XL 1.0.3, PhysioNet, DOI 10.13026/kfzx-aw45. Follow the dataset's license. The repo bundles trained weights, not the dataset.

## How to run it

```bash
cd CheckYourHeartHealth/backend
pip install -r requirements.txt
streamlit run app.py
```

The first time you open the risk page, the app trains the model itself (a few seconds) and saves it as `src/risk_prediction/heart_risk_model.joblib`. To train and re-measure by hand:

```bash
python src/risk_prediction/train_model.py
```

That writes the model file and `src/risk_prediction/metrics.json`.

Secrets go in your own `.env` file (not committed). The app reads them with python-dotenv.
