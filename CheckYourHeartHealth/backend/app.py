import html
import time
# app.py  (Streamlit wrapper for Transformer Heart Disease Prediction with GA & PSO)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import altair as alt
import io
import os
import sqlite3
from dotenv import load_dotenv
import sys
import hashlib
import time
import json
import tempfile
import socket
import socketserver
import threading
import http.server
from functools import partial
from pathlib import Path
from PIL import Image
import copy
from src.styling import apply_global_styles as apply_custom_styles
from src.database import (
    init_database, 
    add_user, 
    get_user_by_username, 
    save_assessment, 
    get_unresolved_alerts_for_user, 
    get_unresolved_alerts_for_doctor, 
    get_unresolved_alerts_for_admin, 
    insert_alert,
    resolve_alert, 
    get_user_by_id,
    authenticate_user,
    get_patients_for_doctor,
    get_patient_history,
    add_patient_note,
    get_patient_notes,
    link_patient_to_doctor,
    get_doctors_for_patient,
    get_all_doctor_patient_links,
    unlink_doctor_patient,
    add_link_by_ids,
    get_all_users,
    update_user_role,
    set_user_active_status,
    get_system_counts,
    get_recent_assessments,
    get_all_alerts,
    log_audit_action,
    get_recent_audit_logs,
    delete_user,
    add_user,
    get_most_critical_alert_for_doctor,
    acknowledge_alert,
    save_explainability,
    get_latest_explainability,
    get_explainability_trend,
    save_ecg_assessment,
    get_ecg_assessments_for_user,
    get_ecg_assessments_for_doctor,
    get_recent_ecg_assessments,
    set_user_notify_doctor,
    get_latest_assessment_id,
    add_lifestyle_plan,
    get_lifestyle_plans,
    update_lifestyle_plan,
    add_telehealth_message,
    get_telehealth_messages,
)
try:
    from src.ecg_classifier import ecg_classifier as ecg_analyzer
except Exception as _exc:
    ecg_analyzer = None
    ECG_IMPORT_ERROR = str(_exc)
    ECG_AVAILABLE = False
import io
import base64
from datetime import datetime

# Dataset path
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'src', 'risk_prediction', 'cardio_train_clean.csv')

# 3D Heart viewer assets
HEART_VIEWER_DIR = (Path(__file__).parent / "heart-master").resolve()
_HEART_VIEWER_SERVER_KEY = "heart_viewer_static_server"


@st.cache_data(show_spinner=False)
def _load_dataset(path: str) -> pd.DataFrame:
    """Load the cardiovascular dataset, cached across reruns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def _start_heart_viewer_server() -> int:
    """Start or reuse a local HTTP server that serves the 3D heart assets."""
    if not HEART_VIEWER_DIR.exists():
        raise FileNotFoundError(f"3D assets not found at {HEART_VIEWER_DIR}")

    if _HEART_VIEWER_SERVER_KEY in st.session_state:
        return st.session_state[_HEART_VIEWER_SERVER_KEY]["port"]

    handler = partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(HEART_VIEWER_DIR),
    )
    socketserver.TCPServer.allow_reuse_address = True

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_sock:
        temp_sock.bind(("127.0.0.1", 0))
        host, port = temp_sock.getsockname()

    httpd = socketserver.TCPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    st.session_state[_HEART_VIEWER_SERVER_KEY] = {
        "httpd": httpd,
        "thread": thread,
        "port": port,
    }
    return port


def _render_3d_viewer_page():
    from src.heart_viewer import render_heart_viewer
    render_heart_viewer(HEART_VIEWER_DIR)


# The signal model loads lazily when a valid ECG is uploaded.
ECG_AVAILABLE = ecg_analyzer is not None
ECG_IMPORT_ERROR = None if ECG_AVAILABLE else globals().get('ECG_IMPORT_ERROR', 'ECG import failed')
ECG_CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
ECG_PAGE_LABEL = 'ECG Classification'
ECG_REVIEW_LABEL = 'ECG Review'
THREE_D_PAGE_LABEL = '3D Heart Viewer'
GA_FEATURE_ANALYSIS_LABEL = 'GA Feature Analysis'
ECG_ANALYSIS_LABEL = 'ECG Analysis'

load_dotenv()


# Trained risk model (LightGBM, see src/risk_prediction/train_model.py)
_RISK_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'src', 'risk_prediction', 'heart_risk_model.joblib')

_ENSEMBLE_PATH = os.path.join(os.path.dirname(__file__), 'src', 'risk_prediction', 'heart_risk_ensemble.joblib')

@st.cache_resource(show_spinner="Loading the risk model (the LightGBM fallback trains itself on a very first run, about 20 seconds)...")
def _load_risk_model():
    """Prefer the trained stacking ensemble; fall back to the single LightGBM model."""
    import joblib
    try:
        from src.risk_prediction.ensemble import StackedRiskModel
        if os.path.exists(_ENSEMBLE_PATH):
            artifact = joblib.load(_ENSEMBLE_PATH)
        else:
            import hashlib
            from pathlib import Path
            folder = Path(_ENSEMBLE_PATH).parent
            manifest = json.loads((folder / 'ensemble_manifest.json').read_text())
            encoded = ''.join((folder / f).read_text().strip() for f in manifest['files'])
            raw = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
                raise ValueError('Ensemble weight checksum mismatch')
            artifact = joblib.load(io.BytesIO(raw))
        return {"kind": "ensemble", "model": StackedRiskModel(artifact)}
    except Exception as exc:
        print(f"[Ensemble unavailable, using LightGBM] {exc}")
    if not os.path.exists(_RISK_MODEL_PATH):
        from src.risk_prediction import train_model
        train_model.main()
    return {"kind": "lightgbm", **joblib.load(_RISK_MODEL_PATH)}

def _model_risk_percent(user_data):
    """Risk (0-100) from the trained model. Raises an error if neither trained model can be loaded."""
    try:
        bundle = _load_risk_model()
        bmi = user_data.get("bmi", user_data.get("BMI"))
        data = {**user_data, "bmi": bmi}
        if bundle["kind"] == "ensemble":
            model = bundle["model"]
            row = pd.DataFrame([data])[model.features]
            return float(model.predict_proba(row)[0]) * 100.0, "Stacked ensemble (LightGBM + Random Forest + Transformer)"
        row = pd.DataFrame([data])[bundle["features"]]
        return float(bundle["model"].predict_proba(row)[0, 1]) * 100.0, "LightGBM (trained on the cardio dataset)"
    except Exception as exc:
        raise RuntimeError("No risk model is available. No score was produced.") from exc

# Utility helpers
def safe_log_action(admin_id, action, details=""):
    if not admin_id:
        return
    try:
        log_audit_action(admin_id, action, details)
    except Exception as exc:
        print(f"[Audit log skipped] {action}: {details} -> {exc}")

# Initialize the database
init_database()


def _seed_demo_users():
    # Throwaway demo accounts so visitors can try each role. They hold no real
    # data and are recreated automatically on a fresh database.
    demo_users = [
        ('patient-demo', 'DemoPatient2026!', 'Demo Patient', 'patient'),
        ('doctor-demo', 'DemoDoctor2026!', 'Demo Doctor', 'doctor'),
        ('admin-demo', 'DemoAdmin2026!', 'Demo Admin', 'admin'),
    ]
    for username, password, name, role in demo_users:
        try:
            if not get_user_by_username(username):
                add_user(username, password, name, role)
        except Exception:
            pass


_seed_demo_users()

# Load only the model actually used by predictions. Resource-cached across reruns.
PREDICTOR = None
PREDICTOR_ERROR = None
try:
    PREDICTOR = _load_risk_model()
except Exception as exc:
    PREDICTOR_ERROR = str(exc)

def _determine_risk_category(score: float):
    if score >= 40:
        return "High Risk", "High cardiovascular risk detected. Please consult a doctor immediately."
    return "Low Risk", "Maintain healthy habits and consider regular check-ups with your doctor."


def _compute_rule_based_risk(user_data):
    """Compute a rule-based cardiovascular risk score (0-100) using global-style thresholds.

    This function uses age bands, blood pressure, cholesterol, glucose/diabetes,
    BMI, smoking, alcohol use, and physical activity (type + hours) to build an
    interpretable risk score aligned with the plan described in the app.
    """
    risk = 5.0  # baseline

    def _to_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _to_int(val, default=0):
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    age = _to_float(user_data.get("age", 0))
    gender_code = user_data.get("gender")
    gender = "male" if gender_code == 2 else "female"

    # Gender/age adjustment: higher baseline for males and older/middle-aged women
    if gender_code == 2:
        risk += 7.242
    elif gender_code == 1 and age >= 45:
        risk += 4.13

    ap_hi = _to_float(user_data.get("ap_hi", 0))
    ap_lo = _to_float(user_data.get("ap_lo", 0))
    chol = _to_int(user_data.get("cholesterol", 1), 1)
    gluc = _to_int(user_data.get("gluc", 1), 1)
    diabetes = _to_int(user_data.get("diabetes", 0), 0)
    bmi = _to_float(user_data.get("BMI", user_data.get("bmi", 0)))
    smoke = _to_int(user_data.get("smoke", 0), 0)
    alco = _to_int(user_data.get("alco", 0), 0)
    active = _to_int(user_data.get("active", 0), 0)
    activity_type = user_data.get("activity_type")
    activity_hours = _to_float(user_data.get("activity_hours_per_week", 0))

    severe_risk_factors = 0

    # Age bands
    if age >= 75:
        risk += 15.0
    elif age >= 65:
        risk += 12.0
    elif age >= 55:
        risk += 9.0
    elif age >= 45:
        risk += 6.0
    elif age >= 35:
        risk += 3.0

    # Blood pressure stages
    if ap_hi >= 140 or ap_lo >= 90:
        risk += 10.0
        severe_risk_factors += 1
    elif ap_hi >= 130 or ap_lo >= 85:
        risk += 5.0

    # Cholesterol
    if chol == 3:  # High
        risk += 10.0
        severe_risk_factors += 1
    elif chol == 2:  # Borderline
        risk += 5.0

    # Glucose / Diabetes
    if gluc == 3 or diabetes == 2:  # Diabetes
        risk += 10.0
        severe_risk_factors += 1
    elif gluc == 2 or diabetes == 1:  # Prediabetes
        risk += 5.0

    # BMI
    if bmi >= 30:
        risk += 7.0
        severe_risk_factors += 1
    elif bmi >= 25:
        risk += 4.0

    # Smoking
    if smoke == 1:
        risk += 7.0
        severe_risk_factors += 1

    # Alcohol
    if alco == 1:
        risk += 3.0

    # Physical inactivity
    if active == 0 and activity_hours <= 0:
        risk += 5.0
        severe_risk_factors += 1

    # Physical activity reductions (type + duration)
    minutes = activity_hours * 60.0
    if active == 1 and minutes > 0:
        reduction = 0.0
        activity_label = (activity_type or "").lower()
        if minutes >= 150:
            if "yoga" in activity_label:
                reduction = 7.0
            elif "gym" in activity_label or "strength" in activity_label:
                reduction = 6.0
            elif "sport" in activity_label:
                reduction = 6.0
            elif "walk" in activity_label:
                reduction = 5.0
            elif "run" in activity_label or "jog" in activity_label:
                reduction = 6.0
            else:
                reduction = 5.0
        else:
            reduction = 3.0

        risk -= reduction
        risk = max(risk, 5.0)

    # Age × risk-factor interactions
    if age < 45 and severe_risk_factors >= 3:
        risk *= 1.10

    if age >= 70 and severe_risk_factors >= 3:
        risk *= 1.20

    if age >= 70 and severe_risk_factors <= 1:
        risk *= 0.9

    if 35 <= age <= 55 and severe_risk_factors == 0:
        risk = max(risk, 15.0)
        risk = min(risk, 35.0)

    # Clamp to [0, 100]
    risk = max(0.0, min(100.0, risk))
    return risk

def _generate_recommendations(user_data):
    """Generate personalized health recommendations based on user data."""
    recommendations = [
        "Adopt a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins.",
        "Aim for at least 150 minutes of moderate physical activity each week." 
    ]

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    bmi = _to_float(user_data.get("BMI", user_data.get("bmi")))
    if bmi is not None:
        if bmi >= 30:
            recommendations.append(
                "Your BMI indicates obesity. Focus on gradual weight loss with a structured meal plan and regular exercise; consider working with a healthcare professional." 
            )
        elif bmi >= 25:
            recommendations.append(
                "Your BMI is in the overweight range. Prioritize healthy eating patterns and consistent physical activity to reduce weight and heart risk." 
            )

    active_flag = _to_int(user_data.get("active"))
    activity_hours = _to_float(user_data.get("activity_hours_per_week")) or 0.0
    if (active_flag == 0) or activity_hours <= 0:
        recommendations.append(
            "Start incorporating regular activity such as brisk walking most days of the week to build a habit of movement." 
        )
    elif activity_hours < 2.5:
        recommendations.append(
            "Increase your weekly activity towards at least 150 minutes of moderate exercise to strengthen your heart." 
        )

    if _to_int(user_data.get("smoke")) == 1:
        recommendations.append(
            "Quit smoking as soon as possible—it greatly reduces your cardiovascular risk and improves overall health." 
        )

    if _to_int(user_data.get("alco")) == 1:
        recommendations.append(
            "Limit alcohol intake and avoid binge drinking to keep your blood pressure and heart risk under control." 
        )

    systolic = _to_float(user_data.get("ap_hi")) or 0.0
    diastolic = _to_float(user_data.get("ap_lo")) or 0.0
    if systolic >= 140 or diastolic >= 90:
        recommendations.append(
            "Your blood pressure is high—adopt a low-salt diet, stay active, manage stress, and schedule follow-up with your doctor." 
        )
    elif systolic >= 130 or diastolic >= 85:
        recommendations.append(
            "Your blood pressure is elevated; reinforce lifestyle control with reduced sodium, regular exercise, and weight management." 
        )

    cholesterol = _to_int(user_data.get("cholesterol"))
    if cholesterol == 3:
        recommendations.append(
            "Cholesterol is high. Cut back on fried and processed foods, limit red meat, and consult your doctor about management." 
        )
    elif cholesterol == 2:
        recommendations.append(
            "Cholesterol is borderline—reduce fried/processed foods and red meat to help bring levels down." 
        )

    gluc = _to_int(user_data.get("gluc"))
    diabetes_status = _to_int(user_data.get("diabetes"))
    if (gluc is not None and gluc >= 2) or (diabetes_status is not None and diabetes_status >= 1):
        recommendations.append(
            "Manage blood sugar by cutting down on sugary drinks and refined carbs, staying active, and keeping regular follow-up appointments." 
        )

    age = _to_float(user_data.get("age"))
    if age is not None and age >= 40:
        recommendations.append(
            "Schedule regular heart health screenings—blood pressure, cholesterol, and blood sugar checks are important from age 40 onward." 
        )

    if len(recommendations) == 2:
        recommendations.append(
            "Continue your healthy habits and schedule routine check-ups to stay on track." 
        )

    return recommendations


def _generate_history_advice(assessments):
    """Plain do/don't advice built from the patient's own assessment history.

    assessments: rows of (id, ts, input_json, risk_score, risk_category, model_name),
    newest first. Rule-based and honest - it only reflects the numbers the patient
    actually recorded. Not medical advice.
    """
    dos = []
    donts = []
    notes = []

    parsed = []
    for row in assessments:
        try:
            _id, ts, input_json, score, category, _model = row
            data = json.loads(input_json) if input_json else {}
            parsed.append({"ts": ts, "score": score, "category": category, "data": data})
        except Exception:
            continue

    if not parsed:
        return dos, donts, notes

    latest = parsed[0]
    latest_data = latest["data"]
    latest_score = latest["score"]

    def _f(d, key):
        try:
            v = d.get(key)
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Trend across assessments
    if len(parsed) >= 2 and latest_score is not None:
        prev = parsed[1]
        prev_score = prev["score"]
        if prev_score is not None:
            delta = latest_score - prev_score
            if delta >= 5:
                notes.append(
                    f"Your score went up between your last two checks ({prev_score:.1f}% -> {latest_score:.1f}%). "
                    "Don't wait for symptoms - book a check-up and take these results along."
                )
            elif delta <= -5:
                dos.append(
                    f"Your score came down between your last two checks ({prev_score:.1f}% -> {latest_score:.1f}%). "
                    "Keep doing whatever changed - consistency is what keeps it down."
                )

        first = parsed[-1]
        if first["score"] is not None and len(parsed) >= 3:
            overall = latest_score - first["score"]
            if overall >= 10:
                notes.append(
                    f"Across your {len(parsed)} assessments the score has risen by {overall:.0f} points overall. "
                    "That is a pattern worth discussing with a doctor, not a one-off."
                )
            elif overall <= -10:
                dos.append(
                    f"Across your {len(parsed)} assessments your score has dropped by {abs(overall):.0f} points. "
                    "Your longer-term trend is moving the right way - protect it."
                )

    # Factors present in the majority of recorded assessments
    def majority(flag_fn):
        flags = [flag_fn(p["data"]) for p in parsed]
        flags = [f for f in flags if f is not None]
        if not flags:
            return False
        return sum(1 for f in flags if f) > len(flags) / 2

    if majority(lambda d: (_f(d, "ap_hi") or 0) >= 130 or (_f(d, "ap_lo") or 0) >= 85):
        donts.append("Don't add extra salt to food and go easy on pickles, papads and packaged snacks - your blood pressure readings have been on the high side in most of your assessments.")
        dos.append("Check your blood pressure at roughly the same time of day, seated and rested, so your readings stay comparable over time.")
    if majority(lambda d: (_f(d, "smoke") or 0) >= 1):
        donts.append("Don't smoke - in your assessments it is one of the strongest drivers of a higher score. Quitting is the single biggest change available to you.")
        dos.append("If quitting outright is hard, ask your doctor about a quit plan or nicotine replacement - it works better than willpower alone.")
    if majority(lambda d: (_f(d, "alco") or 0) >= 1):
        donts.append("Don't treat alcohol as harmless - your recorded scores were higher when alcohol use was marked. Cut down and keep several alcohol-free days each week.")
    if majority(lambda d: (_f(d, "cholesterol") or 1) >= 2):
        donts.append("Don't lean on fried food, red meat and ghee-heavy cooking - your cholesterol category has been above the desirable range in most of your checks.")
        dos.append("Swap in more vegetables, whole grains, dal and nuts - small daily swaps show up in cholesterol within weeks.")
    if majority(lambda d: (_f(d, "gluc") or 1) >= 2 or (_f(d, "diabetes") or 0) >= 1):
        donts.append("Don't skip blood-sugar follow-ups - your glucose readings have been above the normal range. Sugary drinks and refined carbs are the first things to cut.")
    if majority(lambda d: (_f(d, "BMI") or _f(d, "bmi") or 0) >= 25):
        dos.append("Your recorded BMI has been above the healthy range in most assessments - even 5% weight loss measurably lowers blood pressure and blood sugar.")
        donts.append("Don't chase crash diets - slow, steady changes are the ones that stay off.")
    if majority(lambda d: (_f(d, "activity_hours_per_week") or 0) < 2.5):
        dos.append("Build up to about 150 minutes of brisk walking or similar activity per week - your logged activity has been below that in most assessments.")

    # Changes between the last two assessments
    if len(parsed) >= 2:
        prev_data = parsed[1]["data"]
        bp_now = _f(latest_data, "ap_hi")
        bp_prev = _f(prev_data, "ap_hi")
        if bp_now is not None and bp_prev is not None and bp_now - bp_prev >= 10:
            notes.append(f"Your systolic blood pressure rose from {bp_prev:.0f} to {bp_now:.0f} between your last two assessments.")
        bmi_now = _f(latest_data, "BMI") or _f(latest_data, "bmi")
        bmi_prev = _f(prev_data, "BMI") or _f(prev_data, "bmi")
        if bmi_now is not None and bmi_prev is not None and bmi_prev - bmi_now >= 1:
            dos.append("Your BMI improved between your last two checks - whatever you changed, it is working.")

    return dos, donts, notes


def _render_history_advice(assessments):
    """Render the history-based advice block (shared by patient and doctor views)."""
    dos, donts, notes = _generate_history_advice(assessments)
    if not dos and not donts and not notes:
        st.caption("Record a few assessments and plain, history-based advice will appear here.")
        return
    st.caption("Rule-based pointers from the recorded numbers - not a diagnosis and not a substitute for a doctor.")
    for n in notes:
        st.info(n)
    col_do, col_dont = st.columns(2)
    with col_do:
        st.markdown("**Do**")
        for d in dos:
            st.markdown(f"- {d}")
    with col_dont:
        st.markdown("**Don't**")
        for d in donts:
            st.markdown(f"- {d}")


def _evaluate_what_if_scenarios(user_data, baseline_risk):
    """Re-run the trained model with one change at a time (weight, smoking, blood pressure...) to see how the score moves."""
    scenarios = []

    def add_scenario(title, adjustments, rationale, actions):
        scenario_payload = copy.deepcopy(user_data)
        scenario_payload.update(adjustments)
        try:
            _h = float(scenario_payload.get("height") or 0)
            _w = float(scenario_payload.get("weight") or 0)
            if _h > 0 and _w > 0:
                scenario_payload["bmi"] = scenario_payload["BMI"] = _w / (_h / 100) ** 2
        except (TypeError, ValueError):
            pass
        simulated_risk, _ = _model_risk_percent(scenario_payload)
        delta = simulated_risk - baseline_risk
        scenarios.append(
            {
                "title": title,
                "new_risk": simulated_risk,
                "delta": delta,
                "rationale": rationale,
                "actions": actions,
            }
        )

    def as_float(value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def as_int(value, default=None):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    # Smoking cessation
    if as_int(user_data.get("smoke"), 0) == 1:
        add_scenario(
            "Quit Smoking",
            {"smoke": 0},
            "Nicotine drives arterial damage and keeps your baseline risk elevated.",
            ["Set a quit date and use nicotine replacement or prescription aids.", "Avoid second-hand smoke and track cravings in a journal."],
        )

    # Blood pressure optimisation
    systolic = as_float(user_data.get("ap_hi"), 0)
    diastolic = as_float(user_data.get("ap_lo"), 0)
    if systolic >= 130 or diastolic >= 85:
        add_scenario(
            "Lower Blood Pressure",
            {"ap_hi": max(120.0, systolic - 15.0), "ap_lo": max(75.0, diastolic - 10.0)},
            "Bringing blood pressure back into the normal range relieves stress on artery walls.",
            ["Adopt the DASH diet with less than 1,500 mg of sodium per day.", "Incorporate 30 minutes of brisk activity on at least 5 days per week.", "Discuss antihypertensive medication adjustments with your doctor if already prescribed."],
        )

    # Cholesterol improvement
    cholesterol_level = as_int(user_data.get("cholesterol"), 1)
    if cholesterol_level and cholesterol_level > 1:
        add_scenario(
            "Improve Cholesterol",
            {"cholesterol": max(1, cholesterol_level - 1)},
            "Reducing LDL cholesterol limits plaque formation inside arteries.",
            ["Swap fried foods for baked or grilled alternatives.", "Increase soluble fibre (oats, legumes) and omega-3 rich foods.", "Follow up on lipid panels within 3 months to monitor progress."],
        )

    # Blood sugar / diabetes control
    glucose_level = as_int(user_data.get("gluc"), 1)
    diabetes_status = as_int(user_data.get("diabetes"), 0)
    if (glucose_level and glucose_level > 1) or (diabetes_status and diabetes_status > 0):
        add_scenario(
            "Tighten Glucose Control",
            {"gluc": 1, "diabetes": 0},
            "Improved glycaemic control slows damage to blood vessels and nerves.",
            ["Limit refined carbohydrates and opt for whole grains.", "Schedule routine HbA1c checks and adhere to prescribed medications.", "Pair meals with light activity such as a 10-minute walk to moderate post-meal glucose spikes."],
        )

    # Weight management / BMI reduction
    bmi_value = as_float(user_data.get("BMI", user_data.get("bmi")))
    height_cm = as_float(user_data.get("height"), None)
    if bmi_value and bmi_value >= 27 and height_cm:
        height_m = height_cm / 100.0
        target_bmi = max(22.0, bmi_value - 3.0)
        target_weight = target_bmi * (height_m ** 2)
        add_scenario(
            "Lose 10-15 lbs",
            {
                "BMI": target_bmi,
                "bmi": target_bmi,
                "weight": target_weight,
            },
            "Moderate weight loss improves blood pressure, cholesterol, and insulin sensitivity in parallel.",
            ["Reduce liquid calories (sugary beverages) and prioritise lean proteins and vegetables.", "Aim for a 500 kcal daily deficit through diet and added movement.", "Track weight weekly to stay accountable."],
        )

    # Physical activity increase
    activity_hours = as_float(user_data.get("activity_hours_per_week"), 0.0) or 0.0
    active_flag = as_int(user_data.get("active"), 0)
    if active_flag == 0 or activity_hours < 2.5:
        add_scenario(
            "Reach 150 minutes of activity",
            {
                "active": 1,
                "activity_hours_per_week": max(3.0, activity_hours + 1.5),
                "activity_type": user_data.get("activity_type") or "Walking",
            },
            "Regular aerobic exercise strengthens the heart muscle and supports healthy blood pressure and weight.",
            ["Schedule 30-minute brisk walks five times per week.", "Blend in resistance training twice weekly to preserve muscle mass.", "Use a wearable or phone reminders to track progress."],
        )

    # Alcohol moderation
    if as_int(user_data.get("alco"), 0) == 1:
        add_scenario(
            "Limit Alcohol",
            {"alco": 0},
            "Cutting back on alcohol helps control blood pressure and triglycerides.",
            ["Follow no-more-than 1 drink per day for women or 2 for men.", "Plan alcohol-free days each week and replace drinks with sparkling water."],
        )

    # Filter scenarios that do not improve risk meaningfully
    filtered = [s for s in scenarios if s["delta"] < -0.2]
    if not filtered:
        return []

    # Sort by greatest risk reduction
    filtered.sort(key=lambda s: s["new_risk"])
    return filtered


def _estimate_feature_contributions(user_data, risk_score):
    contributions = []
    if not user_data or risk_score is None:
        return []

    age = user_data.get("age")
    ap_hi = user_data.get("ap_hi", 0)
    ap_lo = user_data.get("ap_lo", 0)
    chol_level = user_data.get("cholesterol", 1)  # 1=Normal, 2=Above Normal, 3=Well Above Normal
    gluc_level = user_data.get("gluc", 1)
    smoke = user_data.get("smoke", 0)
    alco = user_data.get("alco", 0)
    active = user_data.get("active", 0)
    activity_type = user_data.get("activity_type")
    activity_hours = user_data.get("activity_hours_per_week", 0)
    bmi_val = user_data.get("BMI", user_data.get("bmi"))

    # Check for high cholesterol and BP with specific thresholds
    # The cholesterol value is 1 for Normal, 2 for Above Normal, 3 for Well Above Normal
    high_chol = chol_level > 1
    bp_stage2 = ap_hi >= 160 or ap_lo >= 100
    bp_stage1 = (ap_hi >= 140 or ap_lo >= 90) and not bp_stage2
    high_bp = bp_stage1 or bp_stage2

    # Combined risk factor for high cholesterol and BP
    if high_chol and high_bp:
        contributions.append({
            "feature": "High Cholesterol & Blood Pressure",
            "impact": 25.0,
            "direction": "risk_increase",
            "explanation": "The combination of high cholesterol and elevated blood pressure significantly increases cardiovascular risk as both contribute to arterial damage."
        })
    else:
        # Cholesterol alone
        if high_chol:
            if chol_level == 2:
                impact = 10.0
                level_text = "Above Normal"
            else:
                impact = 18.0
                level_text = "Well Above Normal"
            contributions.append({
                "feature": f"High Cholesterol ({level_text})",
                "impact": impact,
                "direction": "risk_increase",
                "explanation": "Elevated cholesterol levels can lead to plaque buildup in arteries, increasing heart disease risk."
            })

        # Blood pressure contributions
        if high_bp:
            if bp_stage2:
                bp_stage = "Stage 2"
                bp_range = "≥160/100 mmHg"
                bp_impact = 22.0
            else:
                bp_stage = "Stage 1"
                bp_range = "140-159/90-99 mmHg"
                bp_impact = 16.0

            contributions.append({
                "feature": "High Blood Pressure",
                "impact": bp_impact,
                "direction": "risk_increase",
                "explanation": f"{bp_stage} hypertension ({bp_range}) puts extra strain on your heart and arteries."
            })

    # Glucose level
    if gluc_level > 1:
        if gluc_level == 2:
            g_impact = 8.0
            g_text = "Above Normal"
        else:
            g_impact = 14.0
            g_text = "Well Above Normal"
        contributions.append({
            "feature": f"High Blood Sugar ({g_text})",
            "impact": g_impact,
            "direction": "risk_increase",
            "explanation": "Elevated blood sugar can damage blood vessels over time and increase cardiovascular risk."
        })

    # Body mass index
    if bmi_val is not None:
        try:
            bmi = float(bmi_val)
        except (TypeError, ValueError):
            bmi = None

        if bmi is not None:
            if bmi >= 30:
                contributions.append({
                    "feature": f"Obesity (BMI {bmi:.1f})",
                    "impact": 15.0,
                    "direction": "risk_increase",
                    "explanation": "Obesity puts extra strain on the heart and is strongly linked to high blood pressure and diabetes."
                })
            elif bmi >= 25:
                contributions.append({
                    "feature": f"Overweight (BMI {bmi:.1f})",
                    "impact": 9.0,
                    "direction": "risk_increase",
                    "explanation": "Excess body weight increases the workload on your heart and blood vessels."
                })
            elif bmi < 18.5:
                contributions.append({
                    "feature": f"Underweight (BMI {bmi:.1f})",
                    "impact": 5.0,
                    "direction": "risk_increase",
                    "explanation": "Being underweight can be associated with underlying illness and reduced cardiovascular reserve."
                })
            else:
                contributions.append({
                    "feature": f"Healthy BMI (BMI {bmi:.1f})",
                    "impact": 6.0,
                    "direction": "risk_decrease",
                    "explanation": "A BMI in the healthy range helps keep blood pressure, cholesterol and blood sugar under control."
                })

    # Smoking
    if smoke == 1:
        contributions.append({
            "feature": "Smoking",
            "impact": 12.0,
            "direction": "risk_increase",
            "explanation": "Smoking damages blood vessels and can significantly increase your cardiovascular risk."
        })

    # Alcohol use
    if alco == 1:
        contributions.append({
            "feature": "Regular alcohol use",
            "impact": 6.0,
            "direction": "risk_increase",
            "explanation": "Frequent alcohol intake can raise blood pressure and triglycerides, increasing cardiovascular risk."
        })

    # Physical activity
    if active == 1:
        try:
            hours_val = float(activity_hours)
        except (TypeError, ValueError):
            hours_val = 0.0

        if hours_val >= 5:
            impact = 14.0
            feature_label = "High level of regular physical activity"
            if isinstance(activity_type, str) and activity_type:
                explanation = (
                    f"Regular exercise (about {hours_val:.1f} hours per week of {activity_type.lower()}) "
                    "is helping to significantly lower your cardiovascular risk."
                )
            else:
                explanation = (
                    f"Regular exercise (about {hours_val:.1f} hours per week) is helping to significantly "
                    "lower your cardiovascular risk."
                )
            direction = "risk_decrease"
        elif hours_val >= 2:
            impact = 10.0
            feature_label = "Regular physical activity"
            explanation = (
                f"Being physically active for around {hours_val:.1f} hours per week improves blood pressure, "
                "cholesterol and weight, helping to lower your risk."
            )
            direction = "risk_decrease"
        elif hours_val > 0:
            impact = 6.0
            feature_label = "Light physical activity"
            explanation = (
                f"Some physical activity (about {hours_val:.1f} hours per week) is beneficial, but increasing "
                "your exercise duration or intensity could further reduce your cardiovascular risk."
            )
            direction = "risk_decrease"
        else:
            impact = 8.0
            feature_label = "Low physical activity"
            explanation = (
                "Too little physical activity reduces circulation and makes it harder to control weight, "
                "blood pressure and blood sugar, increasing cardiovascular risk."
            )
            direction = "risk_increase"

        contributions.append({
            "feature": feature_label,
            "impact": impact,
            "direction": direction,
            "explanation": explanation,
        })
    else:
        contributions.append({
            "feature": "Low physical activity",
            "impact": 8.0,
            "direction": "risk_increase",
            "explanation": "Too little physical activity reduces circulation and makes it harder to control weight, blood pressure and blood sugar."
        })

    # Age
    if age is not None and age > 40:
        age_impact = (age - 40) * 0.4 + 6
        contributions.append({
            "feature": "Age",
            "impact": min(age_impact, 20.0),
            "direction": "risk_increase",
            "explanation": "Advancing age is a natural risk factor for cardiovascular disease."
        })

    # Ensure we don't exceed 5 key factors
    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)[:5]

    # If no contributions were added, add a baseline explanation
    if not contributions:
        contributions.append({
            "feature": "Baseline Health Profile",
            "impact": max(risk_score / 2, 1),
            "direction": "risk_increase",
            "explanation": "Your risk is based on overall health profile. Maintain healthy lifestyle habits."
        })

    return contributions


def _render_explainability_section(contributions, show_actionable=True):
    st.info('General health context from hand-written rules, not an explanation of this model. No per-factor percentage or SHAP/LIME attribution has been measured.')
    for item in contributions or []:
        st.write(f"**{item.get('feature', 'Factor')}**: {item.get('explanation', '')}")


def _render_explainability_trend(rows):
    st.caption("Per-factor effects have not been measured. The old rule-based attribution chart is intentionally not shown.")


def _generate_patient_pdf_report(patient_name, chart_df, explain_rows, patient_data=None, risk_score=None, risk_category=None):
    """Generate a professional hospital-style PDF report for cardiovascular assessment.
    
    Args:
        patient_name: Patient's full name
        chart_df: DataFrame with historical risk scores
        explain_rows: Explainability data rows
        patient_data: Dictionary containing patient vitals and lifestyle data
        risk_score: Current risk score (0-100)
        risk_category: Risk category (Low/Moderate/High/Critical)
    """
    from datetime import datetime
    import uuid
    
    buf = io.BytesIO()
    
    # Generate unique report ID
    report_id = str(uuid.uuid4())[:8].upper()
    report_date = datetime.now().strftime("%B %d, %Y")
    
    with PdfPages(buf) as pdf:
        # ========================================
        # PAGE 1: COVER PAGE
        # ========================================
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.92, 'CHECKYOURHEARTHEALTH', 
                ha='center', va='top', fontsize=24, fontweight='bold', color='#2C3E50')
        ax.text(0.5, 0.88, 'CARDIOVASCULAR ASSESSMENT REPORT', 
                ha='center', va='top', fontsize=16, color='#34495E')
        
        # Report ID and Date
        ax.text(0.5, 0.83, f'Report ID: {report_id}', 
                ha='center', va='top', fontsize=10, color='#7F8C8D')
        ax.text(0.5, 0.81, f'Generated: {report_date}', 
                ha='center', va='top', fontsize=10, color='#7F8C8D')
        
        # Patient Details Table
        ax.text(0.1, 0.75, 'PATIENT INFORMATION', 
                fontsize=12, fontweight='bold', color='#2C3E50')
        
        patient_info = [
            ['Patient Name:', patient_name],
            ['Age:', patient_data.get('age', 'N/A') if patient_data else 'N/A'],
            ['Gender:', patient_data.get('gender', 'N/A') if patient_data else 'N/A'],
            ['Patient ID:', patient_data.get('user_id', 'N/A') if patient_data else 'N/A'],
            ['Assessment Date:', report_date],
            ['Doctor Assigned:', patient_data.get('doctor_name', 'Not Assigned') if patient_data else 'Not Assigned']
        ]
        
        y_pos = 0.72
        for label, value in patient_info:
            ax.text(0.12, y_pos, label, fontsize=10, color='#34495E')
            ax.text(0.35, y_pos, str(value), fontsize=10, fontweight='bold', color='#2C3E50')
            y_pos -= 0.03
        
        # Confidentiality Notice
        ax.text(0.5, 0.08, 'CONFIDENTIAL MEDICAL REPORT', 
                ha='center', fontsize=9, color='#E74C3C', fontweight='bold')
        ax.text(0.5, 0.05, 'This report contains sensitive medical information and should be handled accordingly.', 
                ha='center', fontsize=8, color='#7F8C8D', style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ========================================
        # PAGE 2: EXECUTIVE SUMMARY
        # ========================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'EXECUTIVE SUMMARY', 
                ha='center', fontsize=16, fontweight='bold', color='#2C3E50')
        
        # Current Risk Score (Large Display)
        current_risk = risk_score if risk_score is not None else (chart_df['risk_score'].iloc[-1] if not chart_df.empty else 0)
        risk_cat = risk_category if risk_category else ('High Risk' if current_risk >= 40 else 'Low Risk')
        
        # Risk box
        risk_color = '#E74C3C' if current_risk >= 40 else '#F39C12' if current_risk >= 25 else '#27AE60'
        ax.add_patch(plt.Rectangle((0.25, 0.82), 0.5, 0.08, 
                                   facecolor=risk_color, alpha=0.2, edgecolor=risk_color, linewidth=2))
        ax.text(0.5, 0.88, f'{current_risk:.1f}%', 
                ha='center', va='center', fontsize=36, fontweight='bold', color=risk_color)
        ax.text(0.5, 0.78, risk_cat.upper(), 
                ha='center', fontsize=14, fontweight='bold', color=risk_color)
        
        # Interpretation
        ax.text(0.1, 0.72, 'Clinical Interpretation:', fontsize=11, fontweight='bold', color='#2C3E50')
        
        if current_risk >= 40:
            interpretation = "This assessment indicates HIGH cardiovascular risk. Immediate medical consultation\nis recommended. Multiple risk factors require urgent attention and lifestyle modifications."
        elif current_risk >= 25:
            interpretation = "This assessment indicates MODERATE cardiovascular risk. Regular monitoring and\nlifestyle improvements are advised. Consult with your healthcare provider."
        else:
            interpretation = "This assessment indicates LOW cardiovascular risk. Continue maintaining healthy\nhabits and schedule routine check-ups with your healthcare provider."
        
        ax.text(0.12, 0.68, interpretation, fontsize=9, color='#34495E', verticalalignment='top')
        
        # Top Risk Contributors
        ax.text(0.1, 0.58, 'Top Risk Contributors:', fontsize=11, fontweight='bold', color='#2C3E50')
        
        y_pos = 0.54
        if patient_data:
            contributors = []
            if patient_data.get('ap_hi', 120) > 140:
                contributors.append(('High Blood Pressure', f"Systolic: {patient_data.get('ap_hi')} mmHg"))
            if patient_data.get('cholesterol', 1) > 2:
                contributors.append(('Elevated Cholesterol', 'Above normal levels'))
            if patient_data.get('smoke', 0) == 1:
                contributors.append(('Smoking Habit', 'Active smoker'))
            if patient_data.get('BMI', 25) > 30:
                contributors.append(('Obesity', f"BMI: {patient_data.get('BMI', 0):.1f}"))
            if patient_data.get('active', 1) == 0:
                contributors.append(('Sedentary Lifestyle', 'No physical activity'))
            
            for icon_text, desc in contributors[:5]:
                ax.text(0.12, y_pos, icon_text, fontsize=9, color='#E74C3C', fontweight='bold')
                ax.text(0.4, y_pos, desc, fontsize=9, color='#34495E')
                y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ========================================
        # PAGE 3: CLINICAL & LIFESTYLE INPUTS
        # ========================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'PATIENT CLINICAL & LIFESTYLE DATA', 
                ha='center', fontsize=14, fontweight='bold', color='#2C3E50')
        
        if patient_data:
            # Vitals Table
            ax.text(0.1, 0.90, 'A. VITALS AND CLINICAL PARAMETERS', 
                    fontsize=11, fontweight='bold', color='#2C3E50')
            
            vitals_data = [
                ['Parameter', 'Value', 'Reference Range', 'Status'],
                ['Systolic BP', f"{patient_data.get('ap_hi', 'N/A')} mmHg", '90-120 mmHg', 
                 'Normal' if patient_data.get('ap_hi', 120) <= 120 else 'Elevated'],
                ['Diastolic BP', f"{patient_data.get('ap_lo', 'N/A')} mmHg", '60-80 mmHg', 
                 'Normal' if patient_data.get('ap_lo', 80) <= 80 else 'Elevated'],
                ['Cholesterol', ['Normal', 'Above Normal', 'Well Above'][patient_data.get('cholesterol', 1)-1], 
                 'Normal', 'Normal' if patient_data.get('cholesterol', 1) == 1 else 'High'],
                ['Glucose', ['Normal', 'Above Normal', 'Well Above'][patient_data.get('gluc', 1)-1], 
                 'Normal', 'Normal' if patient_data.get('gluc', 1) == 1 else 'High'],
                ['BMI', f"{patient_data.get('BMI', 'N/A'):.1f}" if isinstance(patient_data.get('BMI'), (int, float)) else 'N/A', 
                 '18.5-24.9', 'Normal' if patient_data.get('BMI', 25) < 25 else 'Overweight'],
                ['Heart Rate', f"{patient_data.get('heart_rate', 'N/A')} bpm" if patient_data.get('heart_rate') else 'N/A', 
                 '60-100 bpm', 'N/A']
            ]
            
            y_pos = 0.86
            for row in vitals_data:
                x_positions = [0.1, 0.32, 0.52, 0.72]
                for i, cell in enumerate(row):
                    weight = 'bold' if row == vitals_data[0] else 'normal'
                    ax.text(x_positions[i], y_pos, str(cell), fontsize=8, fontweight=weight, color='#2C3E50')
                y_pos -= 0.03
                if row == vitals_data[0]:
                    ax.plot([0.1, 0.9], [y_pos+0.01, y_pos+0.01], 'k-', linewidth=0.5)
            
            # Lifestyle Table
            y_pos -= 0.04
            ax.text(0.1, y_pos, 'B. LIFESTYLE FACTORS', 
                    fontsize=11, fontweight='bold', color='#2C3E50')
            
            lifestyle_data = [
                ['Factor', 'Status'],
                ['Smoking', 'Non-Smoker' if patient_data.get('smoke', 0) == 0 else 'Active Smoker'],
                ['Alcohol Consumption', 'No' if patient_data.get('alco', 0) == 0 else 'Yes'],
                ['Physical Activity', 'Active' if patient_data.get('active', 1) == 1 else 'Sedentary'],
                ['Activity Type', patient_data.get('activity_type', 'Not Specified')],
                ['Activity Hours/Week', f"{patient_data.get('activity_hours_per_week', 0)} hours"]
            ]
            
            y_pos -= 0.03
            for row in lifestyle_data:
                weight = 'bold' if row == lifestyle_data[0] else 'normal'
                ax.text(0.1, y_pos, row[0], fontsize=8, fontweight=weight, color='#2C3E50')
                ax.text(0.4, y_pos, row[1], fontsize=8, fontweight=weight, color='#2C3E50')
                y_pos -= 0.03
                if row == lifestyle_data[0]:
                    ax.plot([0.1, 0.7], [y_pos+0.01, y_pos+0.01], 'k-', linewidth=0.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ========================================
        # PAGE 4: RISK BREAKDOWN & TREND
        # ========================================
        if not chart_df.empty:
            fig = plt.figure(figsize=(8.27, 11.69))
            
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.axis('off')
            ax1.text(0.05, 0.8, 'Research demo only. Not a diagnosis.\nPer-factor effects have not been measured.\nNo SHAP/LIME attribution or clinical calibration is claimed.', fontsize=11)

            # Historical Trend
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.set_title('Risk Trend Over Time', fontsize=12, fontweight='bold', color='#2C3E50', pad=15)
            
            sorted_df = chart_df.sort_values('ts')
            ax2.plot(sorted_df['ts'], sorted_df['risk_score'], 
                    marker='o', color='#3498DB', linewidth=2, markersize=6, 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor='#3498DB')
            ax2.fill_between(sorted_df['ts'], sorted_df['risk_score'], alpha=0.2, color='#3498DB')
            ax2.set_xlabel('Assessment Date', fontsize=10, color='#34495E')
            ax2.set_ylabel('Risk Score (%)', fontsize=10, color='#34495E')
            ax2.grid(True, alpha=0.3, linestyle=':')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            fig.autofmt_xdate()
            
            # Trend Statistics
            avg_risk = sorted_df['risk_score'].mean()
            max_risk = sorted_df['risk_score'].max()
            min_risk = sorted_df['risk_score'].min()
            trend = "Increasing" if len(sorted_df) > 1 and sorted_df['risk_score'].iloc[-1] > sorted_df['risk_score'].iloc[0] else "Stable"
            
            stats_text = f"Current: {current_risk:.1f}% | Average: {avg_risk:.1f}% | Max: {max_risk:.1f}% | Min: {min_risk:.1f}% | Trend: {trend}"
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=9, color='#7F8C8D')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # ========================================
        # PAGE 5: ECG ASSESSMENTS (if available)
        # ========================================
        # Get ECG assessments for this patient
        patient_id = patient_data.get('user_id') if patient_data else None
        if patient_id:
            try:
                from src.database import get_ecg_assessments_for_user
                ecg_assessments = get_ecg_assessments_for_user(patient_id)
                
                if ecg_assessments:
                    fig = plt.figure(figsize=(8.27, 11.69))
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    
                    ax.text(0.5, 0.95, 'ECG ASSESSMENTS', 
                            ha='center', fontsize=14, fontweight='bold', color='#2C3E50')
                    
                    y_pos = 0.90
                    for i, ecg_record in enumerate(ecg_assessments[:5]):  # Show up to 5 most recent
                        ecg_id, assessment_id, ts, predicted_label, confidence, probabilities_json, patient_summary, doctor_summary = ecg_record
                        
                        # Format the timestamp
                        try:
                            formatted_ts = pd.to_datetime(ts).strftime("%B %d, %Y at %I:%M %p")
                        except:
                            formatted_ts = str(ts)
                        
                        # Add ECG record header
                        ax.text(0.1, y_pos, f'ECG Assessment #{i+1}', fontsize=11, fontweight='bold', color='#2C3E50')
                        y_pos -= 0.03
                        
                        # Add details
                        ax.text(0.12, y_pos, f'Date/Time: {formatted_ts}', fontsize=9, color='#34495E')
                        y_pos -= 0.025
                        ax.text(0.12, y_pos, f'Diagnosis: {predicted_label}', fontsize=9, color='#34495E')
                        y_pos -= 0.025
                        ax.text(0.12, y_pos, f'Confidence: {float(confidence)*100:.1f}%', fontsize=9, color='#34495E')
                        y_pos -= 0.035
                        
                        # Add a separator line
                        ax.plot([0.1, 0.9], [y_pos, y_pos], 'k-', linewidth=0.5, alpha=0.3)
                        y_pos -= 0.05
                        
                        # Check if we need a new page
                        if y_pos < 0.2:
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            fig = plt.figure(figsize=(8.27, 11.69))
                            ax = fig.add_subplot(111)
                            ax.axis('off')
                            y_pos = 0.95
                    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                # If there's an error fetching ECG data, we simply skip this section
                pass
        
        # ========================================
        # PAGE 6: RECOMMENDATIONS
        # ========================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'PERSONALIZED CLINICAL RECOMMENDATIONS', 
                ha='center', fontsize=14, fontweight='bold', color='#2C3E50')
        
        # AI Explainability Summary
        ax.text(0.1, 0.90, 'AI MODEL EXPLANATION', fontsize=11, fontweight='bold', color='#2C3E50')
        ai_explanation = f"""The cardiovascular risk prediction model analyzed {len(patient_data) if patient_data else 0} clinical and lifestyle \nparameters to generate this {current_risk:.1f}% risk score. The model uses advanced machine learning \nalgorithms trained on cardiovascular health data to identify patterns associated with heart disease risk. \nKey contributing factors include blood pressure readings, cholesterol levels, lifestyle habits, and \nphysical activity patterns."""
        ax.text(0.12, 0.87, ai_explanation, fontsize=8, color='#34495E', verticalalignment='top')
        
        # Lifestyle Recommendations
        y_pos = 0.72
        ax.text(0.1, y_pos, 'LIFESTYLE RECOMMENDATIONS', fontsize=11, fontweight='bold', color='#2C3E50')
        
        lifestyle_recs = [
            "• Engage in at least 150 minutes of moderate-intensity aerobic activity weekly",
            "• Adopt a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins",
            "• Maintain a healthy body weight (BMI 18.5-24.9)",
            "• Limit sodium intake to less than 2,300 mg per day",
            "• Avoid tobacco products and limit alcohol consumption",
            "• Manage stress through relaxation techniques, meditation, or yoga",
            "• Ensure adequate sleep (7-9 hours per night)",
            "• Monitor blood pressure regularly at home"
        ]
        
        y_pos -= 0.03
        for rec in lifestyle_recs:
            ax.text(0.12, y_pos, rec, fontsize=8, color='#34495E')
            y_pos -= 0.03
        
        # Medical Recommendations
        y_pos -= 0.02
        ax.text(0.1, y_pos, 'MEDICAL RECOMMENDATIONS', fontsize=11, fontweight='bold', color='#2C3E50')
        
        medical_recs = [
            "• Schedule regular check-ups with your primary care physician",
            "• Discuss this report with your healthcare provider",
            "• Consider cardiology consultation if risk score is elevated",
            "• Follow prescribed medication regimens as directed",
            "• Monitor and manage chronic conditions (diabetes, hypertension)",
            "• Conduct periodic lipid panel and metabolic screening"
        ]
        
        y_pos -= 0.03
        for rec in medical_recs:
            ax.text(0.12, y_pos, rec, fontsize=8, color='#34495E')
            y_pos -= 0.03
        
        # Footer
        ax.plot([0.1, 0.9], [0.12, 0.12], 'k-', linewidth=0.5)
        ax.text(0.5, 0.09, 'CONFIDENTIAL MEDICAL REPORT', 
                ha='center', fontsize=9, color='#E74C3C', fontweight='bold')
        ax.text(0.5, 0.06, f'Generated by CheckYourHeartHealth System | Report ID: {report_id}', 
                ha='center', fontsize=8, color='#7F8C8D')
        ax.text(0.5, 0.03, 'This report is for informational purposes only and does not replace professional medical advice.', 
                ha='center', fontsize=7, color='#95A5A6', style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    buf.seek(0)
    return buf.getvalue()

from datetime import datetime
import unicodedata

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# import genetic_selection  # Commented out due to sklearn version compatibility issue

def login_page():
    apply_custom_styles()
    st.title("CheckYourHeartHealth")
    st.caption("A cardiovascular machine-learning study. Sign in to explore with fictional data.")
    
    # Toggle between Login and Register
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                attempts = st.session_state.get('_login_attempts', [])
                attempts = [stamp for stamp in attempts if time.time() - stamp < 60]
                if len(attempts) >= 5:
                    st.error('Too many attempts. Wait a minute and try again.')
                    return
                attempts.append(time.time())
                st.session_state['_login_attempts'] = attempts
                user = authenticate_user(username, password)
                if user:
                    st.session_state["_login_attempts"] = []
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_id = user[0]
                    st.session_state.user_role = user[3]  # role is at index 3
                    st.success(f"Welcome {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with register_tab:
        with st.form("register_form"):
            st.subheader("Create New Account")
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password", 
                                       help="Use at least 12 characters. Use a unique password for this demo.")
            confirm_password = st.text_input("Confirm password", type="password")
            full_name = st.text_input("Full Name")
            email = st.text_input("Email (optional)")
            phone = st.text_input("Phone Number (optional)")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            role = 'patient'
            st.caption('Self-registration creates patient accounts only. An administrator provisions doctor/admin accounts. Use fictional data in this demo.')

            register_button = st.form_submit_button("Register")
            
            if register_button:
                if not (new_username and new_password and full_name):
                    st.error("Please fill in all required fields")
                elif len(new_password) < 12:
                    st.error("Use at least 12 characters")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    try:
                        user_id = add_user(
                            username=new_username,
                            password=new_password,
                            name=full_name,
                            role=role,
                            email=email or None,
                            phone_no=phone or None,
                            age=age if age > 0 else None
                        )
                        st.success("Registration successful! Please log in.")
                        st.session_state.current_tab = "Login"
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))


def show_login_page():
    """Backward-compatible wrapper for older call sites expecting show_login_page."""
    login_page()

def handle_oauth_callback():
    # OAuth callback handling has been removed
    pass

def _assessment_target(key):
    uid = st.session_state.get('user_id')
    if st.session_state.get('user_role', 'patient').lower() == 'doctor':
        patients = get_patients_for_doctor(uid)
        if not patients:
            st.info('No linked patients. A patient can link you from Health Summary, or an administrator can create a link.')
            return None
        options = {p['id']: p.get('name') or p['username'] for p in patients}
        return st.selectbox('Patient for this assessment', list(options), format_func=lambda x: options[x], key=key)
    return uid


def _render_ecg_page(role_lower: str):
    st.header('ECG signal classification')
    target_id = _assessment_target('ecg_patient_target')
    if target_id is None:
        return
    st.warning('Research demo only. These scores are not a diagnosis, and do not rule out heart disease. Do not use this app for urgent symptoms.')
    st.caption('1D-CNN trained on PTB-XL 1.0.3. Official patient-wise folds 1-8 / 9 / 10. Held-out macro AUC: 0.930. This is not 93% accuracy.')
    st.write('Upload a 10-second, 100 Hz, 12-lead signal in millivolts. CSV must have 1000 rows and the lead columns below. Photos and scans are not supported.')
    from src.ecg_classifier import LEADS
    st.code(','.join(LEADS))
    st.caption('Classes: NORM = normal ECG pattern, MI = myocardial infarction pattern, STTC = ST/T change, CD = conduction disturbance, HYP = hypertrophy pattern. Labels can overlap.')
    uploaded = st.file_uploader('12-lead signal CSV', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            frame.columns = [str(c).strip().upper() for c in frame.columns]
            if len(frame) != 1000 or set(frame.columns) != set(LEADS) or len(frame.columns) != 12:
                raise ValueError('CSV must have exactly 1000 rows and 12 named lead columns, with no timestamp or index column.')
            signal = frame[LEADS].to_numpy(dtype='float32')
            st.line_chart(pd.DataFrame({'Lead II (mV)':signal[:,1]}, index=np.arange(1000)/100))
            if st.button('Analyze ECG signal'):
                scores = ecg_analyzer.predict(signal)
                st.subheader('Model scores')
                st.bar_chart(pd.DataFrame({'Score': scores}))
                st.caption('Independent sigmoid scores, not percentages of certainty. They need not add to 1. No clinical threshold or calibration was validated.')
                top = max(scores, key=scores.get)
                uid = target_id
                if uid:
                    save_ecg_assessment(uid, 'Top score: '+top, scores[top], scores, patient_summary='Research-only multi-label CNN scores. Not a diagnosis.')
                    st.success('Saved to your ECG history.')
        except Exception as exc:
            st.error(str(exc))


def _render_risk_history(user_id):
    """Show this patient's saved risk assessments (from the database) as a trend and a table."""
    st.subheader("Your risk history")
    rows = [r for r in get_patient_history(user_id).get("assessments", [])]
    records = []
    for _id, ts, input_json, score, category, model_name in rows:
        try:
            data = json.loads(input_json) if input_json else {}
        except (TypeError, ValueError):
            data = {}
        if isinstance(data, dict) and data.get("type") == "ecg":
            continue
        records.append({
            "When": ts, "Risk %": round(float(score), 1), "Category": category,
            "Age": data.get("age"), "Weight (kg)": data.get("weight"),
            "BP": f"{data.get('ap_hi')}/{data.get('ap_lo')}" if data.get("ap_hi") else None,
        })
    if not records:
        st.info("No saved assessments yet. Calculate your risk above and it will show up here.")
        return
    df = pd.DataFrame(records)
    df["When"] = pd.to_datetime(df["When"], errors="coerce")
    df = df.sort_values("When")
    if len(df) >= 2:
        st.line_chart(df.set_index("When")["Risk %"])
        change = df["Risk %"].iloc[-1] - df["Risk %"].iloc[0]
        st.caption(f"{len(df)} assessments. Change since your first one: {change:+.1f} points.")
    else:
        st.caption("1 assessment so far. Come back after your next check to see a trend.")
    st.dataframe(df.sort_values("When", ascending=False), use_container_width=True, hide_index=True)


def show_risk_prediction_page(predictor_ready: bool):
    st.title("CheckYourHeartHealth")
    target_id = _assessment_target('risk_patient_target')
    if target_id is None:
        return
    st.header("Risk assessment")
    st.caption("Explore how the model responds to health measurements. Held-out accuracy: 73.46%. This is a research score, not a personal medical prognosis.")
    
    # Two-column layout for main content
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        
        with st.form("prediction_form"):
            st.subheader("1. Profile")

            col1_inner, col2_inner = st.columns(2)
            with col1_inner:
                full_name = st.text_input("Full Name", key="name_input")
                email_input = st.text_input("Email", key="email_input")
                phone_no = st.text_input("Phone Number", key="phone_input")

            with col2_inner:
                gender = st.selectbox("Gender", ["Female", "Male"], index=0, key="gender_input")
                age = st.number_input("Age", min_value=18, max_value=100, value=50, step=1, format="%d", key="age_input")

            st.markdown("### 2. Measurements")
            col1_meas, col2_meas = st.columns(2)
            with col1_meas:
                height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170, step=1, format="%d", key="height_input")
            with col2_meas:
                weight = st.number_input("Weight (kg)", min_value=40, max_value=200, value=70, step=1, format="%d", key="weight_input")

            st.markdown("### 3. Blood pressure")
            bp_col1, bp_col2 = st.columns(2)
            with bp_col1:
                ap_hi = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=250, value=120, step=1, format="%d", key="ap_hi_input")
            with bp_col2:
                ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=150, value=80, step=1, format="%d", key="ap_lo_input")

            st.markdown("### Health Indicators")
            col1_health, col2_health = st.columns(2)
            with col1_health:
                cholesterol = st.selectbox("Cholesterol", ["Desirable (< 200 mg/dL)", "Borderline high (200-239 mg/dL)", "High (>= 240 mg/dL)"], index=0, key="chol_input")
                gluc = st.selectbox("Glucose", ["Normal (70-99 mg/dL)", "Prediabetes (100-125 mg/dL)", "Diabetes (>= 126 mg/dL)"], index=0, key="gluc_input")

            with col2_health:
                diabetes_status = st.selectbox("Diabetes status", ["No diabetes (HbA1c < 5.7%)", "Prediabetes (HbA1c 5.7-6.4%)", "Diabetes (HbA1c >= 6.5%)"], index=0, key="diabetes_input")
                smoke = st.selectbox("Smoking", ["No", "Yes"], index=0, key="smoke_input")
                alco = st.selectbox("Alcohol", ["No", "Yes"], index=0, key="alco_input")
                activity_type = st.selectbox(
                    "Type of physical activity",
                    ["Walking", "Running/Jogging", "Gym/Strength Training", "Yoga", "Sports", "Other"],
                    index=0,
                    key="activity_type_input",
                )
                activity_hours_per_week = st.number_input(
                    "Hours per week",
                    min_value=0.0,
                    max_value=168.0,
                    value=3.0,
                    step=0.5,
                    key="activity_hours_input",
                )

            submitted = st.form_submit_button("Calculate Risk")
        
    with col2:
        st.subheader("Before you begin")
        st.write("Use fictional information only. Enter blood pressure as systolic / diastolic and check the measurement units.")
        st.divider()
        st.caption("ABOUT THE MODEL")
        st.write("LightGBM, Random Forest and a small tabular Transformer feed a logistic-regression combiner.")
        st.caption("Test AUC 0.8009 · 13,110 held-out records")
        st.divider()
        st.write("This app does not diagnose disease, measure treatment benefits or support emergency decisions.")

    # Handle form submission
    if submitted:
        if "user_id" not in st.session_state:
            st.error("Please log in to save your assessment")
            return

        current_user_id = target_id

        user = get_user_by_username(st.session_state.username)
        if user:
            update_fields = {}
            if full_name and (not user[2] or user[2] != full_name):
                update_fields["name"] = full_name
            if email_input and (not user[6] or user[6] != email_input):
                update_fields["email"] = email_input
            if phone_no and (not user[7] or user[7] != phone_no):
                update_fields["phone_no"] = phone_no
            if age and (not user[8] or user[8] != age):
                update_fields["age"] = age

            if update_fields:
                try:
                    conn = sqlite3.connect("heart_health.db")
                    cursor = conn.cursor()
                    set_clause = ", ".join(f"{k} = ?" for k in update_fields)
                    values = list(update_fields.values()) + [st.session_state.username]
                    cursor.execute(f"UPDATE users SET {set_clause} WHERE username = ?", values)
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.warning(f"Couldn't update profile: {str(e)}")

        if not predictor_ready:
            st.error("Prediction model is unavailable. Please try again later.")
            return

        bmi_value = round(weight / ((height / 100) ** 2), 2) if height > 0 else 0
        pulse_pressure = ap_hi - ap_lo
        map_pressure = ap_lo + (ap_hi - ap_lo) / 3

        if bmi_value < 18.5:
            bmi_category = 0
        elif bmi_value < 25:
            bmi_category = 1
        elif bmi_value < 30:
            bmi_category = 2
        elif bmi_value < 40:
            bmi_category = 3
        else:
            bmi_category = 4

        if age < 35:
            age_group = 0
        elif age < 45:
            age_group = 1
        elif age < 55:
            age_group = 2
        elif age < 65:
            age_group = 3
        else:
            age_group = 4

        if ap_hi < 130 and ap_lo < 80:
            bp_stage = 0
        elif ap_hi < 140 and ap_lo < 90:
            bp_stage = 1
        elif ap_hi < 160 and ap_lo < 100:
            bp_stage = 2
        else:
            bp_stage = 3

        chol_numeric = {"Desirable (< 200 mg/dL)": 1, "Borderline high (200-239 mg/dL)": 2, "High (>= 240 mg/dL)": 3}[cholesterol]
        gluc_numeric = {"Normal (70-99 mg/dL)": 1, "Prediabetes (100-125 mg/dL)": 2, "Diabetes (>= 126 mg/dL)": 3}[gluc]
        diabetes_numeric = {"No diabetes (HbA1c < 5.7%)": 0, "Prediabetes (HbA1c 5.7-6.4%)": 1, "Diabetes (HbA1c >= 6.5%)": 2}[diabetes_status]
        gender_numeric = 2 if gender == "Male" else 1

        user_data = {
            "age": age,
            "gender": gender_numeric,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": chol_numeric,
            "gluc": gluc_numeric,
            "smoke": 1 if smoke == "Yes" else 0,
            "alco": 1 if alco == "Yes" else 0,
            "active": 1 if activity_hours_per_week > 0 else 0,
            "activity_type": activity_type,
            "activity_hours_per_week": activity_hours_per_week,
            "diabetes": diabetes_numeric,
            "BMI": bmi_value,
            "bmi": bmi_value,
            "pulse_pressure": pulse_pressure,
            "MAP": map_pressure,
            "bmi_category": bmi_category,
            "age_group": age_group,
            "bp_stage": bp_stage,
            "age_bmi": age * bmi_value,
            "chol_gluc": chol_numeric * gluc_numeric,
            "bp_interaction": ap_hi * ap_lo,
            "age_bp": age * ap_hi,
            "bmi_bp": bmi_value * ap_hi,
        }

        with st.spinner("Calculating cardiovascular risk..."):
            try:
                risk_score, model_label = _model_risk_percent(user_data)
                risk_category, risk_message = _determine_risk_category(risk_score)

                assessment_id = save_assessment(
                    user_id=current_user_id,
                    risk_score=risk_score,
                    risk_category=risk_category,
                    model_name=model_label,
                    input_json=json.dumps(user_data),
                )

                if risk_score >= 40:
                    conn = sqlite3.connect("heart_health.db")
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO alerts (user_id, assessment_id, type, message, resolved)
                        VALUES (?, (SELECT MAX(id) FROM assessments WHERE user_id = ?), ?, ?, 0)
                        """,
                        (
                            current_user_id,
                            current_user_id,
                            "high_risk",
                            f"High cardiovascular risk detected: {risk_score:.1f}%",
                        ),
                    )
                    conn.commit()
                    conn.close()
                    
                st.subheader("Assessment result")
                
                # Update risk meter display
                risk_class = "risk-low" if risk_score < 30 else "risk-medium" if risk_score < 60 else "risk-high"
                st.markdown(f"""
                <script>
                // Update risk meter with actual values
                document.querySelector('.risk-percentage').innerHTML = '{risk_score:.1f}%';
                document.querySelector('.risk-category').innerHTML = '{risk_category}';
                document.querySelector('.risk-category').className = 'risk-category {risk_class}';
                </script>
                """, unsafe_allow_html=True)
                
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.metric("Risk Score", f"{risk_score:.1f}%")
                with col_result2:
                    st.metric("Category", risk_category)

                st.progress(int(risk_score))
                st.write(risk_message)

                contributions = _estimate_feature_contributions(user_data, risk_score)
                try:
                    save_explainability(assessment_id, contributions)
                except Exception as exc:
                    st.warning(f"Explainability data could not be saved: {exc}")

                st.subheader("Engineered Features Analysis")
                with st.expander("View Advanced Health Metrics"):
                    eng_col1, eng_col2, eng_col3 = st.columns(3)

                    with eng_col1:
                        st.metric(
                            "Pulse Pressure",
                            f"{pulse_pressure:.0f} mmHg",
                            help="Difference between systolic and diastolic BP",
                        )
                        st.metric(
                            "Mean Arterial Pressure",
                            f"{map_pressure:.1f} mmHg",
                            help="Average blood pressure during cardiac cycle",
                        )
                        st.metric(
                            "BMI Category",
                            ["Underweight", "Normal", "Overweight", "Obese", "Morbid"][bmi_category],
                        )

                    with eng_col2:
                        st.metric(
                            "Age Group",
                            ["<35", "35-44", "45-54", "55-64", "65+"][age_group],
                        )
                        st.metric("BP Stage", ["Normal", "Stage 1", "Stage 2", "Stage 3"][bp_stage])
                        st.metric("Age×BMI Interaction", f"{age * bmi_value:.1f}")

                    with eng_col3:
                        st.metric("Chol×Glucose Risk", f"{chol_numeric * gluc_numeric}")
                        st.metric("BP Interaction", f"{ap_hi * ap_lo:.0f}")
                        st.metric("Age×BP Risk", f"{age * ap_hi:.0f}")

                # Add explainability section after engineered features
                st.subheader("General health context (rule-based)")
                _render_explainability_section(contributions)

                recommendations = _generate_recommendations(user_data)
                if recommendations:
                    st.subheader("Recommendations")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")

                scenarios = _evaluate_what_if_scenarios(user_data, risk_score)
                st.subheader("What-if Analysis")
                if scenarios:
                    st.caption("Hypothetical changes to model inputs only. These are not measured treatment effects or promises about your future health.")
                    for scenario in scenarios[:4]:
                        reduction = -scenario["delta"]
                        card = st.container()
                        with card:
                            top_cols = st.columns([3, 1])
                            with top_cols[0]:
                                st.markdown(f"**{scenario['title']}**")
                                st.write(scenario["rationale"])
                            with top_cols[1]:
                                st.metric(
                                    label="Projected Risk",
                                    value=f"{scenario['new_risk']:.1f}%",
                                    delta=f"-{reduction:.1f} pts",
                                )
                            st.markdown("**Action Plan:**")
                            for action in scenario["actions"]:
                                st.markdown(f"- {action}")
                            st.markdown("---")
                else:
                    st.info("Great work — no major lifestyle tweaks are expected to significantly move your score right now.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Try again or check if models are properly trained.")

    if st.session_state.get("user_id"):
        _render_risk_history(target_id)


def show_ga_feature_analysis_page(df):
    st.header("Genetic Algorithm Feature Selection")
    st.write("Settings (for info)")
    population_size = st.number_input("Population Size", min_value=10, max_value=200, value=50, step=10)
    generations = st.number_input("Generations", min_value=5, max_value=100, value=20, step=5)
    run_sel = st.button("Run Selection")
    st.write("Dataset shape:", df.shape)
    st.download_button(
        "Download previously saved selected CSV (if any)",
        data=b"",
        file_name="cardio_data_with_selected_features.csv",
        mime="text/csv",
        key="dl_empty",
    )

    st.subheader("Data preview")
    st.dataframe(df.head())

    if run_sel:
        with st.spinner("Running feature selection..."):
            try:
                from sklearn.model_selection import StratifiedKFold
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.feature_selection import SelectFromModel

                ga_available = True
                try:
                    from sklearn_genetic import GASearchCV  # type: ignore
                    from sklearn_genetic.space import Integer, Categorical, Continuous  # type: ignore
                except ModuleNotFoundError:
                    ga_available = False
                    st.warning(
                        "Genetic optimizer optional dependency `sklearn-genetic-opt` is not installed. "
                        "Falling back to baseline RandomForest feature selection."
                    )
                except Exception as ga_import_error:
                    ga_available = False
                    st.warning(
                        f"Genetic optimizer could not be loaded ({ga_import_error}). Using baseline feature selection instead."
                    )

                X = df.drop(columns=["cardio"])
                y = df["cardio"]
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)

                ga_selector = None
                if ga_available:
                    param_grid = {
                        "n_estimators": Integer(50, 200),
                        "max_depth": Integer(3, 20),
                        "min_samples_split": Integer(2, 20),
                        "min_samples_leaf": Integer(1, 10),
                    }

                    ga_selector = GASearchCV(
                        estimator=estimator,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        param_grid=param_grid,
                        scoring="accuracy",
                        population_size=int(population_size),
                        generations=int(generations),
                        crossover_probability=0.7,
                        mutation_probability=0.1,
                        n_jobs=1,
                        verbose=True,
                    )

                try:
                    if ga_selector is not None:
                        ga_selector.fit(X, y)

                        best_estimator = ga_selector.best_estimator_
                        selector = SelectFromModel(best_estimator, prefit=True)
                        mask = selector.get_support()
                        selected_cols = X.columns[mask]

                        st.success("Genetic algorithm optimization completed.")
                        st.subheader("Best Parameters Found")
                        st.json(ga_selector.best_params_)
                    else:
                        raise RuntimeError("GA selector unavailable")

                except Exception as ga_error:
                    if ga_selector is not None:
                        st.error(f"Genetic algorithm encountered an error: {ga_error}")
                    st.info("Falling back to basic feature selection with SelectFromModel.")

                    selector = SelectFromModel(estimator, prefit=False)
                    selector.fit(X, y)
                    mask = selector.get_support()
                    selected_cols = X.columns[mask]

                    best_estimator = estimator
                    best_estimator.fit(X[selected_cols], y)

                st.subheader("Selected Features")
                st.write(list(selected_cols))

                X_selected = X[selected_cols]
                if "ga_selector" in locals() and hasattr(ga_selector, "best_params_"):
                    final_model = RandomForestClassifier(**ga_selector.best_params_, random_state=42)
                else:
                    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
                final_model.fit(X_selected, y)

                fi = pd.DataFrame(
                    {"feature": selected_cols, "importance": final_model.feature_importances_}
                ).sort_values("importance", ascending=False)
                st.subheader("Feature Importance (top)")
                st.table(fi.head(10))

                baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
                baseline_model.fit(X, y)
                baseline_score = baseline_model.score(X, y)
                selected_score = final_model.score(X_selected, y)

                st.subheader("Performance Comparison")
                st.write(f"Baseline (all features) accuracy: {baseline_score:.4f}")
                st.write(f"Optimized model accuracy: {selected_score:.4f}")
                st.write(f"Improvement: {selected_score - baseline_score:+.4f}")
                csv_buf = io.StringIO()
                df_selected = df[selected_cols.to_list() + ["cardio"]]
                df_selected.to_csv(csv_buf, index=False)
                st.download_button(
                    "Download CSV with selected features",
                    data=csv_buf.getvalue(),
                    file_name="cardio_data_with_selected_features.csv",
                    mime="text/csv",
                )
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 2, 1)
                sns.barplot(data=fi.head(10), x="importance", y="feature")
                plt.title("Top 10 Feature Importance")
                plt.subplot(2, 2, 2)
                feature_counts = [X.shape[1], len(selected_cols)]
                labels = ["Original Features", "Selected Features"]
                colors = ["#ff7f0e", "#2ca02c"]
                plt.pie(
                    feature_counts,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                plt.title("Feature Reduction")
                plt.subplot(2, 2, 3)
                comp_df = pd.DataFrame(
                    {"Model": ["All Features", "Selected Features"], "Accuracy": [baseline_score, selected_score]}
                )
                sns.barplot(data=comp_df, x="Model", y="Accuracy")
                plt.title("Model Performance Comparison")
                plt.subplot(2, 2, 4)
                corr = X_selected.corr()
                sns.heatmap(
                    corr,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    fmt=".2f",
                    cbar_kws={"shrink": 0.8},
                )
                plt.title("Selected Features Correlation")
                plt.tight_layout()
                plt.savefig("genetic_algorithm_results.png", dpi=300, bbox_inches="tight")
                st.image("genetic_algorithm_results.png", use_container_width=True)
            except Exception as e:
                st.error(f"Feature selection failed: {e}")

def main_app():
    st.caption('Educational demo. Use fictional information only. Not for diagnosis, treatment, or emergencies.')
    current = get_user_by_id(st.session_state.get('user_id'))
    if not current or not current[9]:
        st.session_state.clear()
        st.rerun()
    st.session_state.user_role = current[3]
    try:
        df = _load_dataset(DATASET_PATH)
    except Exception as dataset_err:
        st.error(f"Failed to load dataset: {dataset_err}")
        df = pd.DataFrame()
    predictor_ready = PREDICTOR is not None and PREDICTOR_ERROR is None
    # Apply custom styles
    apply_custom_styles()
    
    # Sidebar: user profile with role badge and navigation
    with st.sidebar:
        st.markdown(
            """
            <style>
                .user-profile {
                    background-color: #262730;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                }
                .user-profile-header {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    margin-bottom: 0.5rem;
                }
                .user-avatar {
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.4rem;
                    font-weight: 700;
                    color: #ffffff;
                    background-color: #ff4b4b;
                }
                .user-name {
                    font-weight: 600;
                    font-size: 1.05rem;
                    color: #ffffff;
                    margin-bottom: 0.15rem;
                }
                .user-username {
                    font-size: 0.85rem;
                    color: #aaaaaa;
                }
                .role-badge {
                    background-color: #ff4b4b;
                    color: white;
                    padding: 4px 10px;
                    border-radius: 999px;
                    font-weight: 600;
                    font-size: 0.8rem;
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    margin-top: 0.25rem;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        role = st.session_state.get("user_role", "patient")
        role_lower = role.lower()
        username = st.session_state.get("username", "User")
        user = get_user_by_username(username)

        display_name = username
        if user and user[2]:
            display_name = html.escape(user[2])

        initials = html.escape((user[2] if user and user[2] else username)[0].upper()) if display_name else "U"
        role_icons = {"doctor": "", "patient": "", "admin": ""}
        role_icon = role_icons.get(role_lower, "")

        st.markdown(
            f"""
            <div class="user-profile">
                <div class="user-profile-header">
                    <div class="user-avatar">{initials}</div>
                    <div>
                        <div class="user-name">{display_name}</div>
                        <div class="user-username">@{html.escape(username)}</div>
                        <div class="role-badge">{role_icon} {role.capitalize()}</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if user:
            if user[6]:  # email
                st.markdown(f"Email: {user[6]}")
            if user[7]:  # phone
                st.markdown(f"Phone: {user[7]}")
            if user[8]:  # age
                st.markdown(f"{user[8]} years old")

        st.markdown("</div>", unsafe_allow_html=True)

        # Main navigation section

        # Admins see a focused admin dashboard navigation; patients/doctors see
        # the full experience.
        if role_lower == "admin":
            nav_pages = [
                "Overview",           # Admin system dashboard
                "Alerts",             # Alert center
                ECG_REVIEW_LABEL,
            ]
        else:
            # Base navigation for patients and doctors
            nav_pages = [
                "Risk Prediction",
                ECG_PAGE_LABEL,              # ECG Classification (upload image)
                GA_FEATURE_ANALYSIS_LABEL,   # GA feature exploration
                "Alerts",
                "Health Summary",
                "Overview",
                THREE_D_PAGE_LABEL,          # 3D Heart Viewer
                "Lifestyle Coach",
                "Progress & Badges",
            ]

            # Doctor-specific ECG tools (analysis + review)
            if role_lower == "doctor":
                if ECG_ANALYSIS_LABEL not in nav_pages:
                    nav_pages.insert(2, ECG_ANALYSIS_LABEL)
                if ECG_REVIEW_LABEL not in nav_pages:
                    nav_pages.insert(3, ECG_REVIEW_LABEL)

        default_page = st.session_state.get("nav_page", nav_pages[0])
        if default_page not in nav_pages:
            default_page = nav_pages[0]

        # Wrap navigation in a custom container for richer styling/animation
        st.markdown('<div class="nav-card"><h2>Navigation</h2>', unsafe_allow_html=True)
        page = st.radio(
            "Go to",
            nav_pages,
            index=nav_pages.index(default_page),
            label_visibility="collapsed",
            key="nav_page",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
    
    if page == "Risk Prediction":
        show_risk_prediction_page(predictor_ready)

    elif page == ECG_ANALYSIS_LABEL and role_lower in ('doctor', 'admin'):
        _render_ecg_page(role_lower)
        return

    if page == GA_FEATURE_ANALYSIS_LABEL:
        show_ga_feature_analysis_page(df)
        return

    if page == ECG_REVIEW_LABEL and role_lower in ("doctor", "admin"):
        st.header("ECG Review & Collaboration")
        doctor_id = st.session_state.get("user_id")
        if not doctor_id:
            st.error("Doctor identification missing. Please log in again.")
            return

        st.caption("Review recent ECG classifications, add clinical notes, and share feedback with patients.")

        assessments = get_recent_ecg_assessments(full=True) if role_lower == "admin" else get_ecg_assessments_for_doctor(doctor_id)
        if not assessments:
            st.info("No ECG assessments available for your patients yet.")
        else:
            for record in assessments:
                (
                    ecg_id,
                    assessment_id,
                    ts,
                    predicted_label,
                    confidence,
                    probabilities_json,
                    patient_summary,
                    doctor_summary,
                    patient_id,
                    patient_name,
                    patient_username,
                ) = record

                patient_display = patient_name or patient_username or f"Patient #{patient_id}"
                with st.expander(f"{patient_display} — {predicted_label} ({float(confidence)*100:.1f}% model score) — {ts}"):
                    try:
                        probabilities_map = json.loads(probabilities_json)
                    except Exception:
                        probabilities_map = {}

                    if probabilities_map:
                        prob_df = pd.DataFrame(
                            [(cls, val) for cls, val in probabilities_map.items()],
                            columns=["Class", "Probability"],
                        )
                        st.subheader("Probability Breakdown")
                        st.table(prob_df)

                    if doctor_summary:
                        try:
                            doctor_summary_dict = json.loads(doctor_summary)
                        except Exception:
                            doctor_summary_dict = doctor_summary
                    else:
                        doctor_summary_dict = None

                    if doctor_summary_dict:
                        st.subheader("Explainability Snapshot")
                        clinical_context = doctor_summary_dict.get("clinical_context")
                        if clinical_context:
                            st.write(f"**Clinical Context:** {clinical_context}")
                        shap_plot_path = doctor_summary_dict.get("shap_plot")
                        if shap_plot_path and os.path.exists(shap_plot_path):
                            st.image(shap_plot_path, caption="SHAP Feature Importance", use_container_width=True)
                        feature_importance = doctor_summary_dict.get("feature_importance")
                        if isinstance(feature_importance, dict) and feature_importance:
                            fi_df = pd.DataFrame(
                                list(feature_importance.items()),
                                columns=["Feature", "Importance"],
                            ).head(10)
                            st.table(fi_df)
                        limitations = doctor_summary_dict.get("model_limitations")
                        if limitations:
                            st.caption(f"Model Notes: {limitations}")

                    st.subheader("Patient-Facing Summary")
                    if patient_summary:
                        try:
                            patient_summary_dict = json.loads(patient_summary)
                        except Exception:
                            patient_summary_dict = patient_summary

                        if isinstance(patient_summary_dict, dict):
                            summary_text = patient_summary_dict.get("summary")
                            if summary_text:
                                st.write(summary_text)
                            guidance_text = patient_summary_dict.get("guidance")
                            if guidance_text:
                                st.info(guidance_text)
                            probability_bar = patient_summary_dict.get("probability_bar")
                            if probability_bar and os.path.exists(probability_bar):
                                st.image(probability_bar, caption="Probability Bar", use_container_width=True)
                        else:
                            st.write(patient_summary_dict)
                    else:
                        st.caption("No patient summary stored for this assessment.")

                    st.subheader("Clinical Collaboration")
                    note_cols = st.columns([3, 1])
                    with note_cols[0]:
                        note_key = f"doctor_note_{ecg_id}"
                        note_content = st.text_area(
                            "Add a note for this patient",
                            key=note_key,
                            placeholder="Document follow-up actions, observations, or reminders",
                        )
                    with note_cols[1]:
                        if st.button(
                            "Save Note",
                            key=f"save_note_{ecg_id}",
                            use_container_width=True,
                            disabled=not note_content.strip(),
                        ):
                            try:
                                add_patient_note(doctor_id, patient_id, note_content.strip())
                                st.success("Note saved.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not save note: {exc}")

                    existing_notes = get_patient_notes(doctor_id, patient_id)
                    if existing_notes:
                        st.markdown("**Recent Notes**")
                        for note_text, created_at in existing_notes:
                            st.markdown(
                                f"- *{created_at}* — {note_text}"
                            )
                    else:
                        st.caption("No prior notes recorded for this patient.")

    if page == "Alerts":
        st.header("Alerts")
        user_role = st.session_state.user_role
        user = get_user_by_username(st.session_state.username)
        user_id = user[0] if user else None

        if not user_id:
            st.error("User not found")
            return

        # Critical alert banner for doctors
        if user_role == "doctor":
            critical_alert = get_most_critical_alert_for_doctor(user_id)
            if critical_alert:
                alert_id, patient_id, assessment_id, timestamp, alert_type, message, patient_username, patient_name = critical_alert
                patient_display = patient_name if patient_name else patient_username
                
                # Get risk score and ECG details for the assessment
                risk_score_info = ""
                ecg_info = ""
                if assessment_id:
                    try:
                        conn = sqlite3.connect('heart_health.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT risk_score, input_json
                            FROM assessments
                            WHERE id = ?
                        ''', (assessment_id,))
                        risk_row = cursor.fetchone()
                        conn.close()
                        if risk_row and risk_row[0] is not None:
                            risk_score_info = f"<p><strong>Risk Score:</strong> {risk_row[0]:.1f}%</p>"
                        
                        # Check if this is an ECG assessment
                        if risk_row and risk_row[1]:
                            try:
                                input_data = json.loads(risk_row[1])
                                if isinstance(input_data, dict) and input_data.get('type') == 'ecg':
                                    # Get ECG details
                                    conn = sqlite3.connect('heart_health.db')
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                        SELECT predicted_label, confidence
                                        FROM ecg_assessments
                                        WHERE assessment_id = ?
                                    ''', (assessment_id,))
                                    ecg_row = cursor.fetchone()
                                    conn.close()
                                    if ecg_row:
                                        predicted_label, confidence = ecg_row
                                        ecg_info = f"<p><strong>ECG top model score:</strong> {predicted_label} ({float(confidence)*100:.1f}% model score)</p>"
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                st.markdown(
                    f"<div style='background:#dc3545;color:white;padding:16px;border-radius:10px;margin-bottom:1rem;'>"
                    f"<h4>Demo alert - saved model score</h4>"
                    f"<p><strong>Patient:</strong> {patient_display}</p>"
                    f"<p><strong>Type:</strong> {alert_type.replace('_', ' ').title()}</p>"
                    f"{risk_score_info}"
                    f"{ecg_info}"
                    f"<p><strong>Message:</strong> {message}</p>"
                    f"<p><strong>Time:</strong> {timestamp}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if user_role == "admin":
            st.subheader("All Unresolved Alerts (Admin View)")
            alerts = get_unresolved_alerts_for_admin()
        elif user_role == "doctor":
            st.subheader("Unresolved Alerts for Your Patients (Doctor View)")
            alerts = get_unresolved_alerts_for_doctor(user_id)
        elif user_role == "patient":
            st.subheader("Your Unresolved Alerts (Patient View)")
            alerts = get_unresolved_alerts_for_user(user_id)
            user_rec = get_user_by_id(user_id)
            notify_val = False
            if user_rec and len(user_rec) > 10:
                try:
                    notify_val = bool(user_rec[10])
                except Exception:
                    notify_val = False
            new_notify = st.toggle("Share major improvements with my doctor", value=notify_val)
            try:
                set_user_notify_doctor(user_id, new_notify)
            except Exception:
                pass
            st.caption("Linked Doctors")
            doctors = get_doctors_for_patient(user_id) if user_id else []
            if doctors:
                for d in doctors:
                    st.write(f"Doctor: {d.get('name') or d.get('username')}")
            else:
                st.caption("No linked doctors")
            if st.button("Send test improvement notification"):
                latest_assessment_id = get_latest_assessment_id(user_id)
                if latest_assessment_id:
                    try:
                        insert_alert(user_id, latest_assessment_id, "progress_milestone", "Demo notification only: no measured improvement asserted")
                        st.success("Notification queued")
                    except Exception as exc:
                        st.error(str(exc))
        else:
            alerts = []
            st.warning("Unknown role, cannot display alerts.")

        # Quick summary of message-type alerts in one place
        if alerts:
            message_alerts = [a for a in alerts if a[4] == "patient_message"]
            if message_alerts:
                st.markdown("### New Messages")
                for alert in message_alerts:
                    alert_id, user_id_val, assessment_id, timestamp, alert_type, message = alert
                    user_rec = get_user_by_id(user_id_val)
                    user_label = (
                        user_rec[2]
                        if user_rec and len(user_rec) > 2 and user_rec[2]
                        else (user_rec[1] if user_rec and len(user_rec) > 1 else str(user_id_val))
                    )
                    
                    # Get risk score and ECG details for the assessment
                    risk_info = ""
                    ecg_info = ""
                    if assessment_id:
                        try:
                            conn = sqlite3.connect('heart_health.db')
                            cursor = conn.cursor()
                            cursor.execute('''
                                SELECT risk_score, input_json
                                FROM assessments
                                WHERE id = ?
                            ''', (assessment_id,))
                            risk_row = cursor.fetchone()
                            conn.close()
                            if risk_row and risk_row[0] is not None:
                                risk_info = f" (Risk: {risk_row[0]:.1f}%)"
                            
                            # Check if this is an ECG assessment
                            if risk_row and risk_row[1]:
                                try:
                                    input_data = json.loads(risk_row[1])
                                    if isinstance(input_data, dict) and input_data.get('type') == 'ecg':
                                        # Get ECG details
                                        conn = sqlite3.connect('heart_health.db')
                                        cursor = conn.cursor()
                                        cursor.execute('''
                                            SELECT predicted_label, confidence
                                            FROM ecg_assessments
                                            WHERE assessment_id = ?
                                        ''', (assessment_id,))
                                        ecg_row = cursor.fetchone()
                                        conn.close()
                                        if ecg_row:
                                            predicted_label, confidence = ecg_row
                                            ecg_info = f" (ECG: {predicted_label}, {float(confidence)*100:.1f}% model score)"
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    
                    st.markdown(
                        f"**{user_label}**{risk_info}{ecg_info} — {message}  \n+  <span style='font-size:0.8rem;color:#aaa;'>({timestamp})</span>",
                        unsafe_allow_html=True,
                    )

        if alerts:
            for alert in alerts:
                alert_id, user_id_val, assessment_id, timestamp, alert_type, message = alert
                user_rec = get_user_by_id(user_id_val)
                user_label = (
                    user_rec[2]
                    if user_rec and len(user_rec) > 2 and user_rec[2]
                    else (user_rec[1] if user_rec and len(user_rec) > 1 else str(user_id_val))
                )
                severity_map = {
                    "high_risk": ("High Risk", "#e63946"),
                    "risk_increase": ("Rising Risk", "#f4a261"),
                    "medication": ("Medication", "#ffb703"),
                    "general": ("Info", "#89c2d9"),
                    "reminder": ("Reminder", "#457b9d"),
                    "patient_message": ("Patient Message", "#6a4c93"),
                    "progress_milestone": ("Progress Milestone", "#2a9d8f"),
                }
                severity_label, severity_color = severity_map.get(
                    alert_type, (alert_type.title() if alert_type else "General", "#6c757d")
                )

                # Get assessment details including risk score
                assessment_details = None
                ecg_details = None
                if assessment_id:
                    try:
                        conn = sqlite3.connect('heart_health.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT risk_score, risk_category, ts, input_json
                            FROM assessments
                            WHERE id = ?
                        ''', (assessment_id,))
                        assessment_details = cursor.fetchone()
                        
                        # Check if this is an ECG assessment
                        if assessment_details:
                            input_json = assessment_details[3]
                            if input_json:
                                try:
                                    input_data = json.loads(input_json)
                                    if isinstance(input_data, dict) and input_data.get('type') == 'ecg':
                                        # Get ECG details
                                        cursor.execute('''
                                            SELECT predicted_label, confidence, probabilities_json
                                            FROM ecg_assessments
                                            WHERE assessment_id = ?
                                        ''', (assessment_id,))
                                        ecg_details = cursor.fetchone()
                                except Exception:
                                    pass
                        conn.close()
                    except Exception:
                        assessment_details = None

                # Format the alert title with risk score if available
                alert_title = f"{severity_label}: {message}"
                if assessment_details and assessment_details[0] is not None:
                    risk_score = assessment_details[0]
                    alert_title += f" (Risk: {risk_score:.1f}%)"
                alert_title += f" — Patient: {user_label} — {timestamp}"

                with st.expander(
                    alert_title,
                    expanded=False,
                ):
                    st.markdown(
                        f"<span style='color:{severity_color}; font-weight:600;'>Type:</span> {alert_type or 'general'}",
                        unsafe_allow_html=True,
                    )
                    st.write(message)
                    st.caption(f"Logged at {timestamp}")
                    
                    # Display risk score and category if available
                    if assessment_details:
                        risk_score, risk_category, assessment_ts, input_json = assessment_details
                        if risk_score is not None:
                            st.markdown(f"**Risk Score:** {risk_score:.1f}%")
                        if risk_category:
                            st.markdown(f"**Risk Category:** {risk_category}")
                        st.markdown(f"**Assessment Time:** {assessment_ts}")
                                        
                        # Display ECG details if this is an ECG assessment
                        if ecg_details:
                            predicted_label, confidence, probabilities_json = ecg_details
                            st.markdown("**ECG Assessment Details:**")
                            st.markdown(f"- **Diagnosis:** {predicted_label}")
                            st.markdown(f"- **Confidence:** {float(confidence)*100:.1f}%")
                                            
                            # Display probabilities if available
                            try:
                                probabilities = json.loads(probabilities_json)
                                if isinstance(probabilities, dict):
                                    st.markdown("**Class Probabilities:**")
                                    prob_list = []
                                    for class_name, prob in probabilities.items():
                                        prob_list.append({"Class": class_name, "Probability": f"{float(prob)*100:.1f}%"})
                                    st.table(pd.DataFrame(prob_list))
                            except Exception:
                                pass

                    # Add communication text box for all roles
                    communication_key = f"communication_note_{alert_id}"
                    communication_note = st.text_area(
                        "Communicate with other users about this alert",
                        key=communication_key,
                        placeholder="Share information, ask questions, or provide updates about this alert...",
                    )
                    
                    action_cols = st.columns(4) if user_role == "admin" else st.columns(3)
                    with action_cols[0]:
                        if st.button(
                            f"Resolve Alert {alert_id}",
                            key=f"resolve_alert_{alert_id}",
                            use_container_width=True,
                        ):
                            resolve_alert(alert_id)
                            if (
                                user_role == "doctor"
                                and communication_note.strip()
                                and user_id_val
                            ):
                                add_patient_note(
                                    st.session_state.get("user_id"),
                                    user_id_val,
                                    f"[Alert {alert_id}] {communication_note.strip()}",
                                )
                            st.success(f"Alert {alert_id} resolved")
                            st.rerun()
                    with action_cols[1]:
                        can_send_reminder = bool(communication_note.strip())
                        if st.button(
                            "Send Patient Reminder",
                            key=f"reminder_alert_{alert_id}",
                            use_container_width=True,
                            disabled=not can_send_reminder,
                        ):
                            try:
                                insert_alert(
                                    user_id_val,
                                    assessment_id,
                                    "reminder",
                                    communication_note.strip(),
                                )
                                if user_role == "doctor" and user_id_val:
                                    add_patient_note(
                                        st.session_state.get("user_id"),
                                        user_id_val,
                                        f"[Reminder for alert {alert_id}] {communication_note.strip()}",
                                    )
                                st.success("Reminder sent to patient.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not send reminder: {exc}")
                    with action_cols[2]:
                        can_send_message = bool(communication_note.strip())
                        if st.button(
                            "Send Message",
                            key=f"message_alert_{alert_id}",
                            use_container_width=True,
                            disabled=not can_send_message,
                        ):
                            try:
                                # Create a chat message alert
                                insert_alert(
                                    user_id_val,
                                    assessment_id,
                                    "patient_message",
                                    communication_note.strip(),
                                )
                                st.success("Message sent successfully.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not send message: {exc}")
                    if user_role == "admin":
                        with action_cols[3]:
                            can_send_doctor_msg = bool(communication_note.strip())
                            if st.button(
                                "Notify Doctor",
                                key=f"doctor_msg_alert_{alert_id}",
                                use_container_width=True,
                                disabled=not can_send_doctor_msg,
                            ):
                                try:
                                    insert_alert(
                                        user_id_val,
                                        assessment_id,
                                        "patient_message",
                                        communication_note.strip(),
                                    )
                                    st.success("Message sent to patient and their doctor(s).")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Could not send doctor notification: {exc}")
        else:
            st.info("No unresolved alerts.")

    elif page == "Health Summary":
        st.header("Health Summary")
        if role_lower == "patient":
            patient_id = st.session_state.get("user_id")
            username = st.session_state.get("username")
            user_record = get_user_by_username(username) if username else None
            history = get_patient_history(patient_id) if patient_id else {}
            assessments = history.get("assessments", []) if history else []
            alerts = history.get("alerts", []) if history else []

            # Highlight critical alerts (high risk / sudden risk increase / reminders)
            critical_alerts = [
                a
                for a in alerts
                if len(a) >= 6 and not a[5] and a[3] in ("high_risk", "risk_increase", "reminder")
            ]
            if critical_alerts:
                alert_id, assessment_id, ts, alert_type, message, resolved = critical_alerts[0]
                
                # Get ECG details if this is an ECG assessment
                ecg_info = ""
                if assessment_id:
                    try:
                        conn = sqlite3.connect('heart_health.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT input_json
                            FROM assessments
                            WHERE id = ?
                        ''', (assessment_id,))
                        assessment_row = cursor.fetchone()
                        conn.close()
                        
                        if assessment_row and assessment_row[0]:
                            input_data = json.loads(assessment_row[0])
                            if isinstance(input_data, dict) and input_data.get('type') == 'ecg':
                                # Get ECG details
                                conn = sqlite3.connect('heart_health.db')
                                cursor = conn.cursor()
                                cursor.execute('''
                                    SELECT predicted_label, confidence
                                    FROM ecg_assessments
                                    WHERE assessment_id = ?
                                ''', (assessment_id,))
                                ecg_row = cursor.fetchone()
                                conn.close()
                                if ecg_row:
                                    predicted_label, confidence = ecg_row
                                    ecg_info = f"<br><strong>ECG top model score:</strong> {predicted_label} ({float(confidence)*100:.1f}% model score)"
                    except Exception:
                        pass
                
                st.markdown(
                    f"<div style='background:#e63946;color:white;padding:16px;border-radius:10px;margin-bottom:1rem;'>"
                    f"<h4>Demo score alert</h4>"
                    f"<b>A saved model score triggered a demo rule.</b><br>"
                    f"{message}{ecg_info}<br>"
                    f"<span style='font-size:0.85rem;'>Recorded at {ts}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Add acknowledgment button for patients
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Acknowledge Alert", key=f"acknowledge_{alert_id}"):
                            acknowledge_alert(alert_id)
                            st.success("Alert acknowledged. Please contact your doctor for further guidance.")
                            st.rerun()
                with col2:
                    st.caption("Acknowledging this alert confirms you've seen it and will contact your doctor if needed.")

            latest_score = None
            latest_category = "No data"
            latest_ts = "—"
            if assessments:
                latest_entry = assessments[0]
                _, ts, _, score, category, _ = latest_entry
                latest_score = score
                latest_category = category or "N/A"
                try:
                    latest_ts = pd.to_datetime(ts).strftime("%b %d, %Y %I:%M %p")
                except Exception:
                    latest_ts = ts

            st.subheader("Personal Dashboard")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest Risk Score", f"{latest_score:.1f}%" if latest_score is not None else "--")
            with col2:
                st.metric("Risk Category", latest_category)
            with col3:
                st.metric("Last Assessment", latest_ts)

            if assessments:
                chart_rows = []
                for assessment in assessments:
                    _, ts, input_json, score, category, model = assessment
                    try:
                        parsed_ts = pd.to_datetime(ts)
                    except Exception:
                        parsed_ts = ts
                    chart_rows.append(
                        {
                            "Timestamp": parsed_ts,
                            "Risk Score": score,
                            "Risk Category": category,
                            "Model": model,
                        }
                    )

                trend_df = pd.DataFrame(chart_rows).sort_values("Timestamp")
                st.subheader("My Risk Trend")
                st.line_chart(trend_df.set_index("Timestamp")["Risk Score"])
                
                # Add explainability for latest assessment
                if assessments:
                    latest_assessment_id = assessments[0][0]
                    latest_score = assessments[0][3]
                    latest_input = None
                    try:
                        latest_input = json.loads(assessments[0][2]) if assessments[0][2] else None
                    except Exception:
                        latest_input = None
                    
                    if latest_input:
                        contributions = _estimate_feature_contributions(latest_input, latest_score)
                        with st.expander("General health context (rule-based)", expanded=False):
                            _render_explainability_section(contributions)
                
                st.dataframe(
                    trend_df.tail(10)
                    .rename(columns={"Timestamp": "Recorded At"})
                    .reset_index(drop=True)
                )
                patient_csv_buffer = io.StringIO()
                trend_df.to_csv(patient_csv_buffer, index=False)
                st.download_button(
                    "Download full assessment history",
                    data=patient_csv_buffer.getvalue(),
                    file_name="my_assessments.csv",
                    mime="text/csv",
                )
            else:
                st.info("No assessments recorded yet. Complete a risk prediction to see your dashboard metrics.")

            if assessments:
                st.subheader("Advice from your assessment history")
                _render_history_advice(assessments)

            # Add comprehensive PDF report section
            st.subheader("Comprehensive Health Report")
            st.markdown("Download a complete PDF report of your health history including risk predictions and ECG results.")
            
            # Get ECG assessments for the patient
            ecg_assessments = get_ecg_assessments_for_user(patient_id) if patient_id else []
            
            if assessments or ecg_assessments:
                if st.button("Generate Comprehensive Health Report (PDF)"):
                    try:
                        # Get patient name for the report
                        patient_name = user_record[2] if user_record and user_record[2] else username
                        
                        # Prepare data for the PDF report
                        chart_rows = []
                        latest_input = None
                        for assessment in assessments:
                            _, ts, input_json, score, category, model = assessment
                            try:
                                parsed_ts = pd.to_datetime(ts)
                            except Exception:
                                parsed_ts = ts
                            chart_rows.append(
                                {
                                    "ts": parsed_ts,
                                    "risk_score": score,
                                    "risk_category": category,
                                    "model": model,
                                }
                            )
                            # Get the latest input data for patient information
                            if input_json and latest_input is None:
                                try:
                                    latest_input = json.loads(input_json)
                                except Exception:
                                    latest_input = None
                        
                        chart_df = pd.DataFrame(chart_rows)
                        if not chart_df.empty:
                            chart_df = chart_df.sort_values("ts")
                        
                        # Get explainability data
                        explain_rows = get_explainability_trend(patient_id) if patient_id else []
                        
                        # Generate the comprehensive PDF report
                        pdf_bytes = _generate_patient_pdf_report(
                            patient_name, 
                            chart_df, 
                            explain_rows,
                            patient_data=latest_input,
                            risk_score=latest_score if latest_score is not None else (chart_df['risk_score'].iloc[-1] if not chart_df.empty else None)
                        )
                        
                        st.download_button(
                            "Download Comprehensive Health Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"{patient_name}_comprehensive_health_report.pdf",
                            mime="application/pdf",
                        )
                    except Exception as pdf_exc:
                        st.error(f"Could not generate PDF report: {pdf_exc}")
            else:
                st.info("No health data available yet. Complete assessments to generate a report.")

            st.subheader("Profile & Preferences")
            with st.expander("Edit profile information", expanded=False):
                if user_record:
                    with st.form("patient_profile_form"):
                        name_val = st.text_input("Full Name", value=user_record[2] or "")
                        email_val = st.text_input("Email", value=user_record[6] or "")
                        phone_val = st.text_input("Phone", value=user_record[7] or "")
                        age_val = st.number_input(
                            "Age",
                            min_value=1,
                            max_value=120,
                            value=int(user_record[8]) if user_record[8] else 30,
                        )
                        submitted = st.form_submit_button("Save profile")
                        if submitted:
                            try:
                                conn = sqlite3.connect('heart_health.db')
                                cursor = conn.cursor()
                                cursor.execute(
                                    "UPDATE users SET name = ?, email = ?, phone_no = ?, age = ? WHERE id = ?",
                                    (name_val, email_val, phone_val, age_val, user_record[0]),
                                )
                                conn.commit()
                                conn.close()
                                st.success("Profile updated.")
                            except Exception as exc:
                                st.error(f"Could not update profile: {exc}")
                else:
                    st.info("Profile data unavailable.")
            st.markdown("### Linked Doctors")
            linked_doctors = get_doctors_for_patient(patient_id) if patient_id else []

            if linked_doctors:
                for doc in linked_doctors:
                    doc_cols = st.columns([3, 1])
                    with doc_cols[0]:
                        st.markdown(
                            f"**Dr. {doc.get('name') or doc.get('username')}**  "+
                            f"Email: {doc.get('email') or 'N/A'}  |  Phone: {doc.get('phone_no') or 'N/A'}"
                        )
                    with doc_cols[1]:
                        if st.button(
                            "Unlink",
                            key=f"unlink_doc_{doc['link_id']}",
                            use_container_width=True,
                        ):
                            unlink_doctor_patient(doc['link_id'])
                            st.success("Doctor unlinked.")
                            st.rerun()

                    # Allow patient to send a reminder / important message to this doctor
                    reminder_key = f"patient_to_doctor_msg_{doc['doctor_id']}"
                    reminder_text = st.text_input(
                        "Send reminder / important message to this doctor",
                        key=reminder_key,
                        placeholder="e.g. Please review my latest assessment or high-risk alert.",
                    )
                    if st.button(
                        f"Send Reminder to Dr. {doc.get('name') or doc.get('username')}",
                        key=f"patient_send_reminder_{doc['doctor_id']}",
                        use_container_width=True,
                        disabled=not reminder_text.strip(),
                    ):
                        try:
                            # Store as a doctor-patient note so the doctor sees it in their view
                            add_patient_note(
                                doc['doctor_id'],
                                patient_id,
                                f"[From patient] {reminder_text.strip()}",
                            )

                            # Also create an alert targeted to this patient so the doctor
                            # sees the reminder in their Alerts view for this patient
                            latest_assessment_id = assessments[0][0] if assessments else None
                            if latest_assessment_id is not None:
                                insert_alert(
                                    patient_id,
                                    latest_assessment_id,
                                    "patient_message",
                                    reminder_text.strip(),
                                )

                            st.success("Reminder sent to your doctor.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not send reminder: {exc}")
            else:
                st.info("No doctors linked yet.")

            st.markdown("#### Add a doctor")
            doctor_username = st.text_input(
                "Doctor username",
                key="patient_link_doctor_username",
                placeholder="Enter the username your doctor uses",
            )
            if st.button("Link Doctor", use_container_width=True):
                if not doctor_username.strip():
                    st.warning("Please enter a doctor username.")
                else:
                    success, msg = link_patient_to_doctor(patient_id, doctor_username.strip())
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        elif role_lower == "doctor":
            doctor_id = st.session_state.get("user_id")
            
            # Critical alert banner for doctors
            critical_alert = get_most_critical_alert_for_doctor(doctor_id)
            if critical_alert:
                alert_id, patient_id, assessment_id, timestamp, alert_type, message, patient_username, patient_name = critical_alert
                patient_display = patient_name if patient_name else patient_username
                st.markdown(
                    f"<div style='background:#dc3545;color:white;padding:16px;border-radius:10px;margin-bottom:1rem;'>"
                    f"<h4>Demo alert - saved model score</h4>"
                    f"<p><strong>Patient:</strong> {patient_display}</p>"
                    f"<p><strong>Type:</strong> {alert_type.replace('_', ' ').title()}</p>"
                    f"<p><strong>Message:</strong> {message}</p>"
                    f"<p><strong>Time:</strong> {timestamp}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.subheader("Summary of your patients' recent assessments")

            patients = get_patients_for_doctor(doctor_id) if doctor_id else []
            if not doctor_id or not patients:
                st.info("No patients linked to your account yet.")
            else:
                summary_rows = []
                for p in patients:
                    name = p.get("name") or p.get("username") or "Unknown"
                    last_ts = p.get("last_assessment_ts")
                    if last_ts:
                        try:
                            last_dt = datetime.fromisoformat(last_ts)
                            last_label = last_dt.strftime("%b %d, %Y %I:%M %p")
                        except ValueError:
                            last_label = last_ts
                    else:
                        last_label = "No assessments yet"

                    summary_rows.append(
                        {
                            "Patient": name,
                            "Age": p.get("age"),
                            "Latest Risk Score": p.get("risk_score"),
                            "Risk Category": p.get("risk_category"),
                            "Last Assessment": last_label,
                            "Unresolved Alerts": p.get("unresolved_alerts", 0),
                        }
                    )

                if summary_rows:
                    df = pd.DataFrame(summary_rows)
                    st.dataframe(df)
                    buf = io.StringIO()
                    df.to_csv(buf, index=False)
                    st.download_button(
                        "Download summary as CSV",
                        data=buf.getvalue(),
                        file_name="doctor_health_summary.csv",
                        mime="text/csv",
                    )
                    
                    # Add comprehensive report section for doctors
                    st.subheader("Patient Comprehensive Reports")
                    st.markdown("Generate and download comprehensive health reports for your patients.")
                    
                    # Patient selection dropdown
                    patient_options = [f"{p.get('name') or p.get('username')} (ID: {p.get('id')})" for p in patients]
                    selected_patient_option = st.selectbox(
                        "Select a patient for comprehensive report",
                        ["-- Select a patient --"] + patient_options,
                        key="doctor_comprehensive_report_select"
                    )
                    
                    if selected_patient_option != "-- Select a patient --":
                        # Extract patient ID from the selected option
                        selected_patient_id = int(selected_patient_option.split("(ID: ")[1].rstrip(")"))
                        selected_patient = next((p for p in patients if p.get("id") == selected_patient_id), None)
                        
                        if selected_patient:
                            patient_name = selected_patient.get("name") or selected_patient.get("username") or "Unknown"
                            
                            # Get patient history
                            history = get_patient_history(selected_patient_id)
                            assessments = history.get("assessments", []) if history else []
                            
                            # Get ECG assessments
                            ecg_assessments = get_ecg_assessments_for_user(selected_patient_id) if selected_patient_id else []
                            
                            if assessments or ecg_assessments:
                                if st.button(f"Generate Comprehensive Report for {patient_name}"):
                                    try:
                                        # Prepare data for the PDF report
                                        chart_rows = []
                                        latest_input = None
                                        for assessment in assessments:
                                            _, ts, input_json, score, category, model = assessment
                                            try:
                                                parsed_ts = pd.to_datetime(ts)
                                            except Exception:
                                                parsed_ts = ts
                                            chart_rows.append(
                                                {
                                                    "ts": parsed_ts,
                                                    "risk_score": score,
                                                    "risk_category": category,
                                                    "model": model,
                                                }
                                            )
                                            # Get the latest input data for patient information
                                            if input_json and latest_input is None:
                                                try:
                                                    latest_input = json.loads(input_json)
                                                except Exception:
                                                    latest_input = None
                                        
                                        chart_df = pd.DataFrame(chart_rows)
                                        if not chart_df.empty:
                                            chart_df = chart_df.sort_values("ts")
                                        
                                        # Get explainability data
                                        explain_rows = get_explainability_trend(selected_patient_id) if selected_patient_id else []
                                        
                                        # Add doctor information to patient data
                                        if latest_input:
                                            latest_input["doctor_name"] = st.session_state.get("username", "Unknown Doctor")
                                        
                                        # Generate the comprehensive PDF report
                                        pdf_bytes = _generate_patient_pdf_report(
                                            patient_name, 
                                            chart_df, 
                                            explain_rows,
                                            patient_data=latest_input,
                                            risk_score=chart_df['risk_score'].iloc[-1] if not chart_df.empty else None
                                        )
                                        
                                        st.download_button(
                                            f"Download {patient_name}'s Comprehensive Health Report (PDF)",
                                            data=pdf_bytes,
                                            file_name=f"{patient_name}_comprehensive_health_report.pdf",
                                            mime="application/pdf",
                                            key=f"doctor_comprehensive_pdf_{selected_patient_id}"
                                        )
                                    except Exception as pdf_exc:
                                        st.error(f"Could not generate PDF report: {pdf_exc}")
                            else:
                                st.info(f"No health data available for {patient_name} yet.")
        elif role_lower == "admin":
            st.header("Admin - Patient Comprehensive Reports")
            st.markdown("Generate and download comprehensive health reports for any patient in the system.")
            
            # Get all users to populate patient list
            all_users = get_all_users()
            patients = [u for u in all_users if u[3] == "patient"]  # Filter for patients only
            
            if not patients:
                st.info("No patients found in the system.")
            else:
                # Create patient options with name and ID
                patient_options = []
                patient_dict = {}
                for patient in patients:
                    patient_id, username, name, role, email, phone, age, active, created_at = patient
                    display_name = name or username or f"Patient #{patient_id}"
                    option_text = f"{display_name} (ID: {patient_id})"
                    patient_options.append(option_text)
                    patient_dict[option_text] = patient_id
                
                # Patient selection dropdown
                selected_patient_option = st.selectbox(
                    "Select a patient for comprehensive report",
                    ["-- Select a patient --"] + patient_options,
                    key="admin_comprehensive_report_select"
                )
                
                if selected_patient_option != "-- Select a patient --":
                    selected_patient_id = patient_dict[selected_patient_option]
                    selected_patient = next((p for p in patients if p[0] == selected_patient_id), None)
                    
                    if selected_patient:
                        patient_id, username, name, role, email, phone, age, active, created_at = selected_patient
                        patient_name = name or username or f"Patient #{patient_id}"
                        
                        # Get patient history
                        history = get_patient_history(selected_patient_id)
                        assessments = history.get("assessments", []) if history else []
                        
                        # Get ECG assessments
                        ecg_assessments = get_ecg_assessments_for_user(selected_patient_id) if selected_patient_id else []
                        
                        if assessments or ecg_assessments:
                            if st.button(f"Generate Comprehensive Report for {patient_name}"):
                                try:
                                    # Prepare data for the PDF report
                                    chart_rows = []
                                    latest_input = None
                                    for assessment in assessments:
                                        _, ts, input_json, score, category, model = assessment
                                        try:
                                            parsed_ts = pd.to_datetime(ts)
                                        except Exception:
                                            parsed_ts = ts
                                        chart_rows.append(
                                            {
                                                "ts": parsed_ts,
                                                "risk_score": score,
                                                "risk_category": category,
                                                "model": model,
                                            }
                                        )
                                        # Get the latest input data for patient information
                                        if input_json and latest_input is None:
                                            try:
                                                latest_input = json.loads(input_json)
                                            except Exception:
                                                latest_input = None
                                    
                                    chart_df = pd.DataFrame(chart_rows)
                                    if not chart_df.empty:
                                        chart_df = chart_df.sort_values("ts")
                                    
                                    # Get explainability data
                                    explain_rows = get_explainability_trend(selected_patient_id) if selected_patient_id else []
                                    
                                    # Add admin information to patient data
                                    if latest_input:
                                        latest_input["doctor_name"] = f"Admin: {st.session_state.get('username', 'System Admin')}"
                                    
                                    # Generate the comprehensive PDF report
                                    pdf_bytes = _generate_patient_pdf_report(
                                        patient_name, 
                                        chart_df, 
                                        explain_rows,
                                        patient_data=latest_input,
                                        risk_score=chart_df['risk_score'].iloc[-1] if not chart_df.empty else None
                                    )
                                    
                                    st.download_button(
                                        f"Download {patient_name}'s Comprehensive Health Report (PDF)",
                                        data=pdf_bytes,
                                        file_name=f"{patient_name}_comprehensive_health_report.pdf",
                                        mime="application/pdf",
                                        key=f"admin_comprehensive_pdf_{selected_patient_id}"
                                    )
                                except Exception as pdf_exc:
                                    st.error(f"Could not generate PDF report: {pdf_exc}")
                        else:
                            st.info(f"No health data available for {patient_name} yet.")

    elif page == "Overview":
        if role_lower == "admin":
            st.markdown('<div class="admin-card"><h2>System Overview</h2>', unsafe_allow_html=True)
            admin_id = st.session_state.get("user_id")
            counts = get_system_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Users", counts["total_users"])
                st.metric("Patients", counts["patients"])
            with col2:
                st.metric("Doctors", counts["doctors"])
                st.metric("Assessments", counts["assessments"])
            with col3:
                st.metric("Admins", counts["admins"])
                st.metric("Unresolved Alerts", counts["unresolved_alerts"])

            latest_by_patient = {}
            for row in get_recent_assessments(50):
                uname = row[1]
                if uname not in latest_by_patient and row[4] is not None:
                    latest_by_patient[uname] = (row[2] or uname, row[4], row[5], row[3])
            high_risk_now = {u: v for u, v in latest_by_patient.items() if v[1] >= 40}
            if high_risk_now:
                st.markdown(f"**Patients currently in the high-risk band (latest assessment): {len(high_risk_now)}**")
                for uname, (pname, score, cat, ts) in sorted(high_risk_now.items(), key=lambda kv: -kv[1][1]):
                    st.write(f"- {pname} (@{uname}): {score:.1f}% ({cat}), recorded {ts}")
            else:
                st.caption("No patients are in the high-risk band on their latest assessment.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="admin-card"><h3>User & Role Management</h3>', unsafe_allow_html=True)
            users = get_all_users()
            with st.form("admin_create_user_form"):
                st.write("Create new account")
                new_username_admin = st.text_input("Username")
                new_password_admin = st.text_input("Password (at least 12 characters)", type="password")
                new_name_admin = st.text_input("Full name")
                new_role_admin = st.selectbox("Role", ["patient", "doctor", "admin"])
                new_email_admin = st.text_input("Email", value="")
                new_phone_admin = st.text_input("Phone", value="")
                new_age_admin = st.number_input("Age", min_value=1, max_value=120, value=30)
                create_user_submit = st.form_submit_button("Add user")
                if create_user_submit:
                    try:
                        new_id = add_user(
                            username=new_username_admin.strip(),
                            password=new_password_admin.strip(),
                            name=new_name_admin.strip() or None,
                            role=new_role_admin,
                            email=new_email_admin.strip() or None,
                            phone_no=new_phone_admin.strip() or None,
                            age=int(new_age_admin),
                        )
                        safe_log_action(admin_id, "create_user", f"{new_username_admin} ({new_role_admin})")
                        st.success("User created.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to create user: {exc}")

            if not users:
                st.info("No users found.")
            else:
                user_df = pd.DataFrame(
                    [
                        {
                            "ID": u[0],
                            "Username": u[1],
                            "Name": u[2],
                            "Role": u[3],
                            "Email": u[4],
                            "Phone": u[5],
                            "Age": u[6],
                            "Active": bool(u[7]),
                            "Created": u[8],
                        }
                        for u in users
                    ]
                )
                csv_buf = io.StringIO()
                user_df.to_csv(csv_buf, index=False)
                st.download_button(
                    "Export users CSV",
                    data=csv_buf.getvalue(),
                    file_name="users.csv",
                    mime="text/csv",
                )

                for user in users:
                    user_id, username_u, name_u, role_u, email_u, phone_u, age_u, active_u, created_at_u = user
                    with st.expander(f"{username_u} ({role_u})", expanded=False):
                        st.write(f"Created: {created_at_u}")
                        with st.form(f"admin_user_form_{user_id}"):
                            new_role = st.selectbox(
                                "Role",
                                ["patient", "doctor", "admin"],
                                index=["patient", "doctor", "admin"].index(role_u),
                                key=f"role_select_{user_id}"
                            )
                            new_active = st.checkbox(
                                "Active",
                                value=bool(active_u),
                                key=f"active_toggle_{user_id}"
                            )
                            new_email = st.text_input("Email", value=email_u or "", key=f"email_admin_{user_id}")
                            new_phone = st.text_input("Phone", value=phone_u or "", key=f"phone_admin_{user_id}")
                            submitted = st.form_submit_button("Save changes")
                            if submitted:
                                updates = []
                                try:
                                    if new_role != role_u:
                                        update_user_role(user_id, new_role)
                                        updates.append(f"role -> {new_role}")
                                    if bool(new_active) != bool(active_u):
                                        set_user_active_status(user_id, new_active)
                                        updates.append(f"active -> {new_active}")
                                    if new_email != (email_u or ""):
                                        conn = sqlite3.connect("heart_health.db")
                                        cursor = conn.cursor()
                                        cursor.execute("UPDATE users SET email = ? WHERE id = ?", (new_email or None, user_id))
                                        conn.commit()
                                        conn.close()
                                        updates.append("email updated")
                                    if new_phone != (phone_u or ""):
                                        conn = sqlite3.connect("heart_health.db")
                                        cursor = conn.cursor()
                                        cursor.execute("UPDATE users SET phone_no = ? WHERE id = ?", (new_phone or None, user_id))
                                        conn.commit()
                                        conn.close()
                                        updates.append("phone updated")
                                    if updates:
                                        safe_log_action(admin_id, "update_user", f"{username_u}: {', '.join(updates)}")
                                        st.success("User updated.")
                                    else:
                                        st.info("No changes made.")
                                except Exception as exc:
                                    st.error(f"Failed to update user: {exc}")

                        # Delete user (admin only; prevent self-delete)
                        delete_disabled = (user_id == admin_id)
                        if st.button(
                            "Delete user",
                            key=f"admin_delete_user_{user_id}",
                            disabled=delete_disabled,
                        ):
                            try:
                                delete_user(user_id)
                                safe_log_action(admin_id, "delete_user", username_u)
                                st.success("User removed.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed to delete user: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="admin-card"><h3>Doctor/Patient Links</h3>', unsafe_allow_html=True)
            with st.form("admin_add_link_form"):
                st.write("Create link")
                doctor_username_input = st.text_input("Doctor username")
                patient_username_input = st.text_input("Patient username")
                add_link_submit = st.form_submit_button("Create link")
                if add_link_submit:
                    success, msg = add_link_by_ids(doctor_username_input.strip(), patient_username_input.strip())
                    if success:
                        safe_log_action(admin_id, "link_doctor_patient", f"{doctor_username_input}->{patient_username_input}")
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            links = get_all_doctor_patient_links()
            if not links:
                st.info("No links recorded yet.")
            else:
                for link in links:
                    link_cols = st.columns([4, 1])
                    with link_cols[0]:
                        st.write(
                            f"Dr. {link['doctor_name'] or link['doctor_username']} ↔ {link['patient_name'] or link['patient_username']}"
                        )
                    with link_cols[1]:
                        if st.button(
                            "Remove",
                            key=f"admin_unlink_{link['link_id']}",
                            use_container_width=True,
                        ):
                            unlink_doctor_patient(link['link_id'])
                            safe_log_action(admin_id, "unlink_doctor_patient", str(link['link_id']))
                            st.success("Link removed")
                            st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="admin-card"><h3>Recent Assessments</h3>', unsafe_allow_html=True)
            recent_assessments = get_recent_assessments(25)
            if recent_assessments:
                assess_df = pd.DataFrame(
                    [
                        {
                            "ID": row[0],
                            "Username": row[1],
                            "Name": row[2],
                            "Timestamp": row[3],
                            "Risk Score": row[4],
                            "Category": row[5],
                            "Model": row[6],
                        }
                        for row in recent_assessments
                    ]
                )
                st.dataframe(assess_df)
                assess_buf = io.StringIO()
                assess_df.to_csv(assess_buf, index=False)
                st.download_button(
                    "Export assessments CSV",
                    data=assess_buf.getvalue(),
                    file_name="recent_assessments.csv",
                    mime="text/csv",
                )
            else:
                st.caption("No assessments recorded.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="admin-card"><h3>Alerts Monitoring</h3>', unsafe_allow_html=True)
            all_alerts = get_all_alerts(100)
            if all_alerts:
                alert_df = pd.DataFrame(
                    [
                        {
                            "ID": row[0],
                            "Username": row[1],
                            "Name": row[2],
                            "Timestamp": row[3],
                            "Type": row[4],
                            "Message": row[5],
                            "Resolved": bool(row[6]),
                        }
                        for row in all_alerts
                    ]
                )
                st.dataframe(alert_df)
                alert_buf = io.StringIO()
                alert_df.to_csv(alert_buf, index=False)
                st.download_button(
                    "Export alerts CSV",
                    data=alert_buf.getvalue(),
                    file_name="alerts.csv",
                    mime="text/csv",
                )
            else:
                st.caption("No alerts logged.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="admin-card"><h3>Audit & Activity Logs</h3>', unsafe_allow_html=True)
            audit_logs = get_recent_audit_logs(50)
            if audit_logs:
                for log in audit_logs:
                    log_id, admin_user, action, details, created_at = log
                    st.write(f"[{created_at}] {admin_user or 'System'} — {action}: {details}")
            else:
                st.caption("No audit entries yet.")
            st.markdown("</div>", unsafe_allow_html=True)
        elif role_lower == "doctor":
            st.header("Your Patients")
            doctor_id = st.session_state.get("user_id")
            patients = get_patients_for_doctor(doctor_id) if doctor_id else []

            if not doctor_id:
                st.info("Doctor ID not found. Please log out and log back in.")
            elif not patients:
                st.info("No patients linked to your account yet.")
            else:
                high_risk_count = sum(1 for p in patients if (p.get("risk_score") or 0) >= 40)
                total_open_alerts = sum((p.get("unresolved_alerts") or 0) for p in patients)
                stat1, stat2, stat3 = st.columns(3)
                stat1.metric("Linked patients", len(patients))
                stat2.metric("High risk (latest assessment)", high_risk_count)
                stat3.metric("Unresolved alerts", total_open_alerts)

                # High-level summary table of patients
                st.subheader("Summary of your patients' recent assessments")
                summary_rows = []
                for p in patients:
                    name = p.get("name") or p.get("username") or "Unknown"
                    last_ts = p.get("last_assessment_ts")
                    if last_ts:
                        try:
                            last_dt = datetime.fromisoformat(last_ts)
                            last_label = last_dt.strftime("%b %d, %Y %I:%M %p")
                        except ValueError:
                            last_label = last_ts
                    else:
                        last_label = "No assessments yet"

                    summary_rows.append(
                        {
                            "Patient": name,
                            "Age": p.get("age"),
                            "Latest Risk Score": p.get("risk_score"),
                            "Risk Category": p.get("risk_category"),
                            "Last Assessment": last_label,
                            "Unresolved Alerts": p.get("unresolved_alerts", 0),
                        }
                    )

                if summary_rows:
                    overview_df = pd.DataFrame(summary_rows)
                    st.dataframe(overview_df)
                    overview_buf = io.StringIO()
                    overview_df.to_csv(overview_buf, index=False)
                    st.download_button(
                        "Download summary as CSV",
                        data=overview_buf.getvalue(),
                        file_name="doctor_overview_summary.csv",
                        mime="text/csv",
                    )

                # Overview-level explainability for a selected patient
                if summary_rows:
                    patient_names = [row["Patient"] for row in summary_rows]
                    selected_name = st.selectbox(
                        "Select a patient for timeline + explainability report",
                        ["-- Select patient --"] + patient_names,
                        key="doctor_overview_patient_select",
                    )

                    if selected_name != "-- Select patient --":
                        selected_patient = next(
                            (
                                p
                                for p in patients
                                if (p.get("name") or p.get("username") or "Unknown") == selected_name
                            ),
                            None,
                        )
                        if selected_patient:
                            selected_id = selected_patient.get("id")
                            history = get_patient_history(selected_id)
                            assmts = history.get("assessments", []) if history else []

                            if assmts:
                                chart_rows = []
                                for assessment in assmts:
                                    _, ts, input_json, score, category, model = assessment
                                    try:
                                        parsed_ts = pd.to_datetime(ts)
                                    except Exception:
                                        parsed_ts = ts
                                    chart_rows.append(
                                        {
                                            "ts": parsed_ts,
                                            "risk_score": score,
                                            "risk_category": category,
                                            "model": model,
                                        }
                                    )

                                chart_df = pd.DataFrame(chart_rows)
                                if not chart_df.empty:
                                    chart_df = chart_df.sort_values("ts")
                                    st.markdown("**Risk score trend (overview report)**")
                                    st.line_chart(chart_df.set_index("ts")["risk_score"])
                                    st.dataframe(
                                        chart_df.tail(5)[["ts", "risk_score", "risk_category", "model"]]
                                        .rename(
                                            columns={
                                                "ts": "Timestamp",
                                                "risk_score": "Score",
                                                "model": "Model",
                                            }
                                        )
                                        .reset_index(drop=True)
                                    )

                                    explain_rows = get_explainability_trend(selected_id)
                                    st.markdown("**Explainability over time (overview report)**")
                                    _render_explainability_trend(explain_rows)

                                    st.markdown("**Advice the patient is seeing (from their history)**")
                                    _render_history_advice(assmts)

                                    try:
                                        pdf_bytes = _generate_patient_pdf_report(
                                            selected_name, chart_df, explain_rows
                                        )
                                        st.download_button(
                                            "Download full PDF report (risk + explainability)",
                                            data=pdf_bytes,
                                            file_name=f"{selected_name}_heart_report.pdf",
                                            mime="application/pdf",
                                            key=f"overview_pdf_{selected_id}",
                                        )
                                    except Exception as pdf_exc:
                                        st.warning(f"Could not generate PDF report: {pdf_exc}")
                            else:
                                st.info("No assessments recorded yet for this patient.")

                st.markdown("---")
                st.subheader("Detailed patient list")

                # Existing detailed list with filters, cards, explainability
                search_query = st.text_input("Search by name or username", key="doctor_patient_search")
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                with col_filter1:
                    risk_filter = st.selectbox(
                        "Risk level",
                        ["All", "High", "Low", "No Data"],
                        key="doctor_patient_risk_filter",
                    )
                with col_filter2:
                    show_alerts_only = st.checkbox(
                        "Only patients with unresolved alerts",
                        value=False,
                        key="doctor_patient_alert_filter",
                    )
                with col_filter3:
                    sort_choice = st.selectbox(
                        "Sort by",
                        ["Latest Assessment", "Highest Risk", "Name"],
                        key="doctor_patient_sort",
                    )

                def determine_level(score):
                    if score is None:
                        return "No Data"
                    if score >= 40:
                        return "High"
                    return "Low"

                filtered_patients = []
                for patient in patients:
                    name = patient.get("name") or patient.get("username") or "Unknown"
                    if search_query:
                        if search_query.lower() not in name.lower() and search_query.lower() not in patient.get("username", "").lower():
                            continue

                    level = determine_level(patient.get("risk_score"))
                    if risk_filter != "All" and level != risk_filter:
                        continue

                    if show_alerts_only and not patient.get("unresolved_alerts"):
                        continue

                    patient_copy = patient.copy()
                    patient_copy["_risk_level"] = level
                    filtered_patients.append(patient_copy)

                if sort_choice == "Highest Risk":
                    filtered_patients.sort(key=lambda p: (p.get("risk_score") is None, -(p.get("risk_score") or 0)))
                elif sort_choice == "Name":
                    filtered_patients.sort(key=lambda p: (p.get("name") or p.get("username") or "").lower())

                if not filtered_patients:
                    st.info("No patients match your filters.")
                    return

                export_df = pd.DataFrame([
                    {
                        "Patient": p.get("name") or p.get("username"),
                        "Age": p.get("age"),
                        "Risk Score": p.get("risk_score"),
                        "Risk Category": p.get("risk_category"),
                        "Last Assessment": p.get("last_assessment_ts"),
                        "Unresolved Alerts": p.get("unresolved_alerts"),
                    }
                    for p in filtered_patients
                ])
                csv_buffer = io.StringIO()
                export_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download patient list (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="doctor_patient_list.csv",
                    mime="text/csv",
                )

                for patient in patients:
                    if patient not in filtered_patients:
                        continue
                    name = patient.get("name") or patient.get("username")
                    age = patient.get("age") or "-"
                    risk_score = patient.get("risk_score")
                    risk_category = patient.get("risk_category") or "No assessment"
                    alerts_count = patient.get("unresolved_alerts", 0)
                    last_ts = patient.get("last_assessment_ts")
                    if last_ts:
                        try:
                            last_dt = datetime.fromisoformat(last_ts)
                            last_label = last_dt.strftime("%b %d, %Y %I:%M %p")
                        except ValueError:
                            last_label = last_ts
                    else:
                        last_label = "No assessments yet"

                    if risk_score is None:
                        risk_display = "—"
                        color = "#6c757d"
                        level = "No Data"
                    elif risk_score >= 40:
                        risk_display = f"{risk_score:.0f}%"
                        color = "#e63946"
                        level = "High"
                    else:
                        risk_display = f"{risk_score:.0f}%"
                        color = "#2a9d8f"
                        level = "Low"

                    alert_badge = f"<span style='background: #3a1d24; color:#ff8a95; padding:2px 8px; border-radius:999px; font-size:0.8rem;'>Alerts: {alerts_count}</span>" if alerts_count else ""

                    st.markdown(
                        f"""
                        <div style="background:#1b2233;border:1px solid #2c3547;border-left:4px solid {color};border-radius:10px;padding:1rem;margin-bottom:1rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div>
                                    <div style="font-size:1.1rem;font-weight:600;color:#e8eaed;">{name}</div>
                                    <div style="color:#9aa4b2;font-size:0.9rem;">Age: {age}</div>
                                    <div style="color:#9aa4b2;font-size:0.9rem;">Last assessment: {last_label}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-size:1.4rem;font-weight:700;color:{color};">{risk_display}</div>
                                    <div style="color:{color};font-weight:600;">{level}</div>
                                    <div style="color:#9aa4b2;font-size:0.85rem;">{risk_category}</div>
                                </div>
                            </div>
                            <div style="margin-top:0.75rem;display:flex;gap:0.5rem;align-items:center;">
                                {alert_badge or ''}
                                <span style="background:#2a3142;color:#c3cbd8;padding:2px 8px;border-radius:999px;font-size:0.8rem;">{last_label}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"View insights for {name}", expanded=False):
                        history = get_patient_history(patient.get("id"))
                        assessments = history.get("assessments", []) if history else []
                        alerts_history = history.get("alerts", []) if history else []
                        patient_id = patient.get("id")

                        if assessments:
                            chart_rows = []
                            latest_input = None
                            for assessment in assessments:
                                _, ts, input_json, score, category, model = assessment
                                try:
                                    parsed_ts = pd.to_datetime(ts)
                                except Exception:
                                    parsed_ts = ts
                                chart_rows.append(
                                    {
                                        "ts": parsed_ts,
                                        "risk_score": score,
                                        "risk_category": category,
                                        "model": model,
                                    }
                                )
                                if input_json and latest_input is None:
                                    try:
                                        latest_input = json.loads(input_json)
                                    except Exception:
                                        latest_input = None

                            chart_df = pd.DataFrame(chart_rows)
                            if not chart_df.empty:
                                chart_df = chart_df.sort_values("ts")
                                st.markdown("**Risk score trend**")
                                st.line_chart(chart_df.set_index("ts")["risk_score"])
                                st.dataframe(
                                    chart_df.tail(5)[["ts", "risk_score", "risk_category", "model"]]
                                    .rename(columns={"ts": "Timestamp", "risk_score": "Score", "model": "Model"})
                                    .reset_index(drop=True)
                                )
                                patient_csv = chart_df.rename(columns={"ts": "Timestamp", "risk_score": "Risk Score"})
                                patient_csv_buffer = io.StringIO()
                                patient_csv.to_csv(patient_csv_buffer, index=False)
                                st.download_button(
                                    "Download assessment history",
                                    data=patient_csv_buffer.getvalue(),
                                    file_name=f"{name}_assessments.csv",
                                    mime="text/csv",
                                    key=f"patient_csv_{patient_id}",
                                )

                                # Explainability trend over time for this patient
                                explain_rows = get_explainability_trend(patient_id)
                                st.markdown("**Explainability over time**")
                                _render_explainability_trend(explain_rows)

                                # PDF report combining risk trend + explainability
                                try:
                                    pdf_bytes = _generate_patient_pdf_report(name, chart_df, explain_rows)
                                    st.download_button(
                                        "Download full PDF report (risk + explainability)",
                                        data=pdf_bytes,
                                        file_name=f"{name}_heart_report.pdf",
                                        mime="application/pdf",
                                        key=f"patient_pdf_{patient_id}",
                                    )
                                except Exception as pdf_exc:
                                    st.warning(f"Could not generate PDF report: {pdf_exc}")

                            if latest_input:
                                st.markdown("**Latest recorded vitals**")
                                vitals_cols = st.columns(3)
                                vitals_cols[0].metric("Systolic BP", latest_input.get("ap_hi", "-"))
                                vitals_cols[0].metric("Diastolic BP", latest_input.get("ap_lo", "-"))
                                vitals_cols[1].metric("Cholesterol", latest_input.get("cholesterol", "-"))
                                vitals_cols[1].metric("Glucose", latest_input.get("gluc", "-"))
                                
                                # Add explainability for doctor review (latest assessment)
                                latest_score = assessments[0][3]
                                contributions = _estimate_feature_contributions(latest_input, latest_score)
                                with st.expander("General health context (rule-based)", expanded=False):
                                    _render_explainability_section(contributions, show_actionable=True)
                                vitals_cols[2].metric("BMI", round(latest_input.get("BMI", latest_input.get("bmi", 0)), 1) if latest_input.get("BMI") or latest_input.get("bmi") else "-")
                                vitals_cols[2].metric("Activity", "Yes" if latest_input.get("active") else "No")
                        else:
                            st.info("No assessments recorded yet for this patient.")

                        if alerts_history:
                            st.markdown("**Alert history**")
                            for alert in alerts_history[:5]:
                                alert_id, assessment_id, ts, alert_type, message, resolved = alert
                                status = "Resolved" if resolved else "Open"
                                st.write(f"- [{status}] {ts}: {message}")
                        else:
                            st.caption("No alerts recorded for this patient.")

                        st.markdown("---")
                        st.markdown("**Doctor notes & follow ups**")
                        existing_notes = get_patient_notes(doctor_id, patient_id) if patient_id else []
                        if existing_notes:
                            for note_text, created_at in existing_notes[:5]:
                                st.write(f"*{created_at}* — {note_text}")
                        else:
                            st.caption("No notes yet for this patient.")

                        note_content = st.text_area(
                            f"Add note for {name}",
                            key=f"patient_note_{patient_id}",
                            placeholder="Summarize observations, follow-up plans, or instructions",
                        )
                        note_cols = st.columns(2)
                        with note_cols[0]:
                            if st.button(
                                f"Save Note ({name})",
                                key=f"save_note_{patient_id}",
                                use_container_width=True,
                                disabled=not bool(note_content.strip()),
                            ):
                                add_patient_note(doctor_id, patient_id, note_content.strip())
                                st.success("Note saved")
                                st.rerun()
                        with note_cols[1]:
                            st.button(
                                "Schedule Follow-up (coming soon)",
                                key=f"followup_btn_{patient_id}",
                                use_container_width=True,
                                disabled=True,
                            )
                        
                        # Lifestyle Coach Plans Management
                        st.markdown("---")
                        st.markdown("Lifestyle Coach Plans")
                        
                        # Get patient's plans
                        patient_plans = get_lifestyle_plans(patient_id)
                        active_patient_plans = [p for p in patient_plans if p[9] == 'active']
                        completed_patient_plans = [p for p in patient_plans if p[9] == 'completed']
                        
                        # Display patient's plan summary
                        plan_cols = st.columns(4)
                        plan_cols[0].metric("Active Plans", len(active_patient_plans))
                        plan_cols[1].metric("Completed", len(completed_patient_plans))
                        best_streak = max([p[12] for p in patient_plans], default=0)
                        plan_cols[2].metric("Best Streak", f"{best_streak} days")
                        total_completions = sum(p[13] for p in patient_plans)
                        plan_cols[3].metric("Total Completions", total_completions)
                        
                        # Display badges earned
                        patient_badges = []
                        if best_streak >= 7:
                            patient_badges.append("Week Warrior")
                        if best_streak >= 30:
                            patient_badges.append("Month Master")
                        if total_completions >= 10:
                            patient_badges.append("10x Achiever")
                        if len(completed_patient_plans) >= 3:
                            patient_badges.append("Plan Finisher")
                        
                        if patient_badges:
                            st.write("**Badges Unlocked:**")
                            for badge in patient_badges:
                                st.success(badge)
                        
                        # Create new plan form
                        with st.form(f"create_plan_{patient_id}"):
                            st.write("**Create New Lifestyle Plan**")
                            plan_name = st.text_input("Plan Name", key=f"plan_name_{patient_id}")
                            plan_desc = st.text_area("Description", key=f"plan_desc_{patient_id}")
                            
                            duration_days = st.number_input("Plan Duration (days)", min_value=1, max_value=365, value=30, help="How many days should this plan last?", key=f"duration_{patient_id}")
                            
                            target_cols = st.columns(3)
                            with target_cols[0]:
                                target_activity = st.number_input("Target Activity (hrs/week)", min_value=0.0, max_value=40.0, value=5.0, key=f"target_act_{patient_id}")
                            with target_cols[1]:
                                target_weight = st.number_input("Target Weight (kg)", min_value=0.0, max_value=200.0, value=70.0, key=f"target_wt_{patient_id}")
                            with target_cols[2]:
                                target_bp = st.text_input("Target BP (e.g., 120/80)", value="120/80", key=f"target_bp_{patient_id}")
                            
                            if st.form_submit_button("Create Plan"):
                                if plan_name:
                                    # Parse BP
                                    try:
                                        bp_parts = target_bp.split("/")
                                        bp_sys = int(bp_parts[0]) if len(bp_parts) > 0 else None
                                        bp_dia = int(bp_parts[1]) if len(bp_parts) > 1 else None
                                    except:
                                        bp_sys, bp_dia = None, None
                                    
                                    add_lifestyle_plan(
                                        user_id=patient_id,
                                        doctor_id=doctor_id,
                                        plan_name=plan_name,
                                        description=plan_desc,
                                        plan_duration_days=duration_days,
                                        target_activity_hours=target_activity,
                                        target_weight=target_weight,
                                        target_bp_systolic=bp_sys,
                                        target_bp_diastolic=bp_dia
                                    )
                                    st.success(f"Created plan '{plan_name}' for {name}!")
                                    st.rerun()
                                else:
                                    st.error("Please provide a plan name")
                        
                        # Display existing plans
                        if patient_plans:
                            st.write("**Existing Plans:**")
                            for plan in patient_plans:
                                plan_id, uid, did, pname, pdesc, tact, twt, tbps, tbpd, pstatus, pnotes, pstreak, plongest, pcomp, plast, pcreated, pupdated = plan
                                
                                status_badge = "" if pstatus == 'active' else "" if pstatus == 'completed' else ""
                                st.write(f"{status_badge} **{pname}** (Streak: {pstreak} days, Completed: {pcomp} times)")
                                if pnotes:
                                    st.caption(f"Latest Note: {pnotes}")
        else:
            st.header("Health Dashboard Overview")

            patient_id = st.session_state.get("user_id")
            history = get_patient_history(patient_id) if patient_id else None
            assessments = history.get("assessments", []) if history else []

            col1, col2, col3 = st.columns(3)

            if assessments:
                latest = assessments[0]
                _, ts, input_json, score, category, model = latest
                current_score = float(score) if score is not None else 0.0

                previous_score = assessments[1][3] if len(assessments) >= 2 else None
                if previous_score is not None:
                    delta_value = current_score - float(previous_score)
                    delta_label = f"{delta_value:+.1f}%"
                else:
                    delta_label = "First reading"

                with col1:
                    st.metric("Current Risk Score", f"{current_score:.1f}%", delta_label)

                try:
                    last_dt = datetime.fromisoformat(ts)
                    last_label = last_dt.strftime("%b %d, %Y")
                    days_ago = (datetime.now() - last_dt).days
                    if days_ago == 0:
                        last_delta = "today"
                    elif days_ago > 0:
                        last_delta = f"{days_ago} days ago"
                    else:
                        last_delta = None
                except Exception:
                    last_label = str(ts)
                    last_delta = None

                with col2:
                    st.metric("Last Check-up", last_label, last_delta)

                with col3:
                    st.metric("Total Assessments", len(assessments))
            else:
                with col1:
                    st.metric("Current Risk Score", "–", None)
                with col2:
                    st.metric("Last Check-up", "No records", None)
                with col3:
                    st.metric("Total Assessments", 0)

            st.subheader("Health Metrics")

            latest_input = None
            if assessments:
                latest = assessments[0]
                _, ts, input_json, score, category, model = latest
                if input_json:
                    try:
                        latest_input = json.loads(input_json)
                    except Exception:
                        latest_input = None

            metrics = {}
            if latest_input:
                ap_hi = latest_input.get("ap_hi")
                ap_lo = latest_input.get("ap_lo")
                if ap_hi is not None and ap_lo is not None:
                    metrics["Blood Pressure"] = f"{ap_hi}/{ap_lo} mmHg"

                chol_value = (
                    latest_input.get("cholesterol_display")
                    or latest_input.get("cholesterol")
                    or latest_input.get("cholesterol_category")
                )
                if chol_value is not None:
                    metrics["Cholesterol"] = str(chol_value)

                gluc_value = (
                    latest_input.get("glucose_display")
                    or latest_input.get("gluc")
                    or latest_input.get("glucose_category")
                )
                if gluc_value is not None:
                    metrics["Glucose"] = str(gluc_value)

            if metrics:
                for metric_name, value in metrics.items():
                    st.metric(metric_name, value)
            else:
                st.info("No recent vitals found. Please complete a risk assessment to see your latest metrics.")

    elif page == ECG_PAGE_LABEL:
        _render_ecg_page(role_lower)

    elif page == THREE_D_PAGE_LABEL:
        _render_3d_viewer_page()

    elif page == "Lifestyle Coach":
        st.header("Lifestyle Coach")
        
        role = st.session_state.get("user_role", "patient")
        role_lower = role.lower()
        user_id = st.session_state.get("user_id")
        
        if not user_id:
            st.warning("Please log in to access Lifestyle Coach.")
            return
        
        # Doctor/Admin View - Manage Patient Plans
        if role_lower in ("doctor", "admin"):
            st.caption("Create and manage lifestyle plans for your patients.")
            
            # Get doctor's patients
            if role_lower == "doctor":
                patients = get_patients_for_doctor(user_id)
            else:
                # Admin can see all patients
                all_users = get_all_users()
                patients = [{
                    'id': u[0],
                    'username': u[1],
                    'name': u[2],
                    'role': u[3]
                } for u in all_users if u[3] == 'patient']
            
            if not patients:
                st.info("No patients linked to your account yet.")
                return
            
            # Select patient
            patient_names = [p.get('name') or p.get('username') for p in patients]
            selected_patient_name = st.selectbox(
                "Select Patient",
                ["-- Select a patient --"] + patient_names,
                key="lifestyle_patient_select"
            )
            
            if selected_patient_name != "-- Select a patient --":
                # Find selected patient
                selected_patient = next(
                    (p for p in patients if (p.get('name') or p.get('username')) == selected_patient_name),
                    None
                )
                
                if selected_patient:
                    patient_id = selected_patient.get('id')
                    st.markdown("---")
                    st.subheader(f"Plans for {selected_patient_name}")
                    
                    # Get patient's plans
                    patient_plans = get_lifestyle_plans(patient_id)
                    active_patient_plans = [p for p in patient_plans if p[9] == 'active']
                    completed_patient_plans = [p for p in patient_plans if p[9] == 'completed']
                    
                    # Display patient's plan summary
                    plan_cols = st.columns(4)
                    plan_cols[0].metric("Active Plans", len(active_patient_plans))
                    plan_cols[1].metric("Completed", len(completed_patient_plans))
                    best_streak = max([p[12] for p in patient_plans], default=0)
                    plan_cols[2].metric("Best Streak", f"{best_streak} days")
                    total_completions = sum(p[13] for p in patient_plans)
                    plan_cols[3].metric("Total Completions", total_completions)
                    
                    # Display badges earned
                    patient_badges = []
                    if best_streak >= 7:
                        patient_badges.append("Week Warrior")
                    if best_streak >= 30:
                        patient_badges.append("Month Master")
                    if total_completions >= 10:
                        patient_badges.append("10x Achiever")
                    if total_completions >= 50:
                        patient_badges.append("Half Century")
                    if len(completed_patient_plans) >= 3:
                        patient_badges.append("Plan Finisher")
                    
                    if patient_badges:
                        st.write("**Patient's Badges:**")
                        badge_cols = st.columns(len(patient_badges))
                        for i, badge in enumerate(patient_badges):
                            badge_cols[i].success(badge)
                    
                    st.markdown("---")
                    
                    # Create new plan form
                    with st.form(f"create_plan_lifestyle_{patient_id}"):
                        st.subheader("Create New Plan")
                        plan_name = st.text_input("Plan Name", placeholder="e.g., 30-Day Fitness Challenge")
                        plan_desc = st.text_area("Description", placeholder="Describe the goals and expectations...")
                        
                        duration_days = st.number_input("Plan Duration (days)", min_value=1, max_value=365, value=30, help="How many days should this plan last?")
                        
                        target_cols = st.columns(3)
                        with target_cols[0]:
                            target_activity = st.number_input("Target Activity (hrs/week)", min_value=0.0, max_value=40.0, value=5.0)
                        with target_cols[1]:
                            target_weight = st.number_input("Target Weight (kg)", min_value=0.0, max_value=200.0, value=70.0)
                        with target_cols[2]:
                            target_bp = st.text_input("Target BP (e.g., 120/80)", value="120/80")
                        
                        if st.form_submit_button("Create Plan", use_container_width=True):
                            if plan_name:
                                # Parse BP
                                try:
                                    bp_parts = target_bp.split("/")
                                    bp_sys = int(bp_parts[0]) if len(bp_parts) > 0 else None
                                    bp_dia = int(bp_parts[1]) if len(bp_parts) > 1 else None
                                except:
                                    bp_sys, bp_dia = None, None
                                
                                add_lifestyle_plan(
                                    user_id=patient_id,
                                    doctor_id=user_id,
                                    plan_name=plan_name,
                                    description=plan_desc,
                                    plan_duration_days=duration_days,
                                    target_activity_hours=target_activity,
                                    target_weight=target_weight,
                                    target_bp_systolic=bp_sys,
                                    target_bp_diastolic=bp_dia
                                )
                                st.success(f"Created plan '{plan_name}' for {selected_patient_name}!")
                                st.rerun()
                            else:
                                st.error("Please provide a plan name")
                    
                    # Display existing plans
                    if patient_plans:
                        st.markdown("---")
                        st.subheader("Existing Plans")
                        
                        for plan in patient_plans:
                            plan_id, uid, did, pname, pdesc, pduration, tact, twt, tbps, tbpd, pstatus, pnotes, pstreak, plongest, pcomp, plast, pcreated, pupdated = plan
                            
                            pstatus = pstatus or 'active'  # Default to 'active' if None
                            status_text = pstatus.upper()
                            
                            with st.expander(f"{pname} - {status_text}"):
                                st.write(f"**Description:** {pdesc or 'No description'}")
                                
                                if pduration:
                                    st.info(f"Duration: {pduration} days")
                                
                                # Display targets
                                if any([tact, twt, tbps]):
                                    st.write("**Targets:**")
                                    target_display_cols = st.columns(3)
                                    if tact:
                                        target_display_cols[0].metric("Activity (hrs/week)", f"{tact}")
                                    if twt:
                                        target_display_cols[1].metric("Weight (kg)", f"{twt}")
                                    if tbps and tbpd:
                                        target_display_cols[2].metric("Blood Pressure", f"{tbps}/{tbpd}")
                                
                                # Progress stats
                                prog_cols = st.columns(4)
                                prog_cols[0].metric("Current Streak", f"{pstreak} days")
                                prog_cols[1].metric("Longest Streak", f"{plongest} days")
                                prog_cols[2].metric("Completions", pcomp)
                                prog_cols[3].metric("Last Update", plast or "Never")
                                
                                # Show plan progress if duration is set
                                if pduration:
                                    progress_percent = min(100, int(pcomp / pduration * 100))
                                    st.progress(progress_percent / 100)
                                    st.caption(f"Progress: {pcomp}/{pduration} days ({progress_percent}%)")
                                
                                if pnotes:
                                    st.info(f"Latest Progress Note: {pnotes}")
                                
                                st.caption(f"Created: {pcreated} | Updated: {pupdated}")
                    else:
                        st.info("No plans created yet for this patient.")
        
        # Patient View - See and Update Their Plans
        else:
            st.caption("Follow your personalized plans and track your progress.")
            
            patient_id = user_id
            
            # Get all plans for this patient
            all_plans = get_lifestyle_plans(patient_id)
            active_plans = [p for p in all_plans if p[9] == 'active']  # status at index 9
            completed_plans = [p for p in all_plans if p[9] == 'completed']
            
            # Calculate gamification stats
            total_streaks = sum(p[11] for p in all_plans)  # streak_count at index 11
            best_streak = max([p[12] for p in all_plans], default=0)  # longest_streak at index 12
            total_completions = sum(p[13] for p in all_plans)  # completed_count at index 13
            
            # Display achievements
            st.subheader("Your Achievements")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Active Plans", len(active_plans))
            with col2:
                st.metric("Best Streak", f"{best_streak} days")
            with col3:
                st.metric("Total Completions", total_completions)
            with col4:
                st.metric("Completed Plans", len(completed_plans))
            
            # Display badges
            badges = []
            if best_streak >= 7:
                badges.append("Week Warrior")
            if best_streak >= 30:
                badges.append("Month Master")
            if total_completions >= 10:
                badges.append("10x Achiever")
            if total_completions >= 50:
                badges.append("Half Century")
            if len(completed_plans) >= 3:
                badges.append("Plan Finisher")
            
            if badges:
                st.write("**Unlocked Badges:**")
                cols = st.columns(len(badges))
                for i, badge in enumerate(badges):
                    cols[i].success(badge)
            
            st.markdown("---")
            
            # Active Plans Section
            if active_plans:
                st.subheader("Active Plans")
                for plan in active_plans:
                    plan_id, user_id, doctor_id, plan_name, description, plan_duration, target_activity, target_weight, target_bp_sys, target_bp_dia, status, progress_notes, streak, longest, completed, last_completion, created_at, updated_at = plan
                    
                    with st.expander(f"{plan_name} (Streak: {streak} days)", expanded=True):
                        st.write(f"**Description:** {description or 'No description'}")
                        
                        if plan_duration:
                            st.info(f"Plan Duration: {plan_duration} days")
                        
                        # Display targets
                        if any([target_activity, target_weight, target_bp_sys]):
                            st.write("**Targets:**")
                            target_cols = st.columns(3)
                            if target_activity:
                                target_cols[0].metric("Activity Hours/Week", f"{target_activity}")
                            if target_weight:
                                target_cols[1].metric("Target Weight (kg)", f"{target_weight}")
                            if target_bp_sys and target_bp_dia:
                                target_cols[2].metric("Target BP", f"{target_bp_sys}/{target_bp_dia}")
                        
                        # Progress stats
                        stats_cols = st.columns(4)
                        stats_cols[0].metric("Current Streak", f"{streak} days")
                        stats_cols[1].metric("Longest Streak", f"{longest} days")
                        stats_cols[2].metric("Completions", completed)
                        stats_cols[3].metric("Last Update", last_completion or "Never")
                        
                        if progress_notes:
                            st.info(f"Progress Notes: {progress_notes}")
                        
                        # Update form
                        with st.form(f"update_plan_{plan_id}"):
                            new_progress = st.text_area("Add Progress Note", key=f"progress_{plan_id}")
                            mark_done = st.checkbox("Mark today's goal as completed", key=f"done_{plan_id}")
                            change_status = st.selectbox("Change Status", ["", "completed", "cancelled"], key=f"status_{plan_id}")
                            
                            # Show plan progress if duration is set
                            if plan_duration:
                                progress_percent = min(100, int((completed + (1 if mark_done else 0)) / plan_duration * 100))
                                st.progress(progress_percent / 100)
                                st.caption(f"Progress: {completed + (1 if mark_done else 0)}/{plan_duration} days ({progress_percent}%)")
                            
                            if st.form_submit_button("Update Progress"):
                                update_lifestyle_plan(
                                    plan_id,
                                    status=change_status if change_status else None,
                                    progress_notes=new_progress if new_progress else None,
                                    mark_completed=mark_done
                                )
                                st.success(f"Updated {plan_name}!")
                                st.rerun()
            else:
                st.info("No active plans yet. Your doctor can create personalized plans for you.")
            
            # Completed Plans Section
            if completed_plans:
                st.markdown("---")
                st.subheader("Completed Plans")
                for plan in completed_plans:
                    plan_id, user_id, doctor_id, plan_name, description, plan_duration, target_activity, target_weight, target_bp_sys, target_bp_dia, status, progress_notes, streak, longest, completed, last_completion, created_at, updated_at = plan
                    
                    with st.expander(f"{plan_name} - Completed"):
                        st.write(f"**Description:** {description or 'No description'}")
                        
                        if plan_duration:
                            st.success(f"Completed {plan_duration}-day plan!")
                        
                        comp_cols = st.columns(3)
                        comp_cols[0].metric("Best Streak", f"{longest} days")
                        comp_cols[1].metric("Total Completions", completed)
                        comp_cols[2].metric("Completed On", last_completion or "N/A")
                        if progress_notes:
                            st.info(f"Final Notes: {progress_notes}")

    elif page == "Progress & Badges":
        st.header("Progress & Badges")
        st.caption("Track your risk trend and earn improvement badges.")
        patient_id = st.session_state.get("user_id")
        history = get_patient_history(patient_id) if patient_id else None
        assessments = history.get("assessments", []) if history else []
        if assessments:
            rows = []
            for a in assessments:
                _, ts, input_json, score, category, model = a
                try:
                    inp = json.loads(input_json) if input_json else {}
                except Exception:
                    inp = {}
                rows.append({
                    "ts": pd.to_datetime(ts),
                    "risk": score,
                    "bmi_cat": inp.get("bmi_category"),
                    "activity_hours": inp.get("activity_hours_per_week", 0.0)
                })
            dfp = pd.DataFrame(rows).sort_values("ts")
            st.line_chart(dfp.set_index("ts")["risk"]) 
            badges = []
            if len(dfp) >= 2 and (dfp.iloc[-2]["risk"] - dfp.iloc[-1]["risk"]) >= 10:
                badges.append("Risk Reducer")
            if len(dfp) >= 2:
                prev_bmi = dfp.iloc[-2]["bmi_cat"]
                latest_bmi = dfp.iloc[-1]["bmi_cat"]
                if isinstance(prev_bmi, int) and isinstance(latest_bmi, int) and latest_bmi < prev_bmi:
                    badges.append("BMI Dropper")
            if dfp["activity_hours"].rolling(3).mean().iloc[-1] and dfp["activity_hours"].rolling(3).mean().iloc[-1] >= 2.5:
                badges.append("Active Streak")
            if badges:
                st.subheader("Badges")
                for b in badges:
                    st.success(f"{b}")
            else:
                st.info("No badges yet. Keep going!")
        else:
            st.info("No data yet")
# Main authentication logic
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
else:
    # Add a logout button in the sidebar
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    main_app()
