"""Updated research views. Legacy role pages share the existing application context."""
import streamlit as st

def render_health_context(contributions, show_actionable=True):
    st.info("General health context from hand-written rules, not an explanation of this model. No per-factor percentage or SHAP/LIME attribution has been measured.")
    for item in contributions or []:
        st.write(f"**{item.get('feature', 'Factor')}**: {item.get('explanation', '')}")

def render_risk_prediction_page(predictor_ready: bool):
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



def install(context):
    # Keep the existing authenticated role workflows and model instance.
    globals().update({key:value for key,value in context.items() if not key.startswith("__")})
    from src.heart_viewer import render_heart_viewer
    from src.styling import apply_global_styles
    context["_render_explainability_section"] = render_health_context
    context["show_risk_prediction_page"] = render_risk_prediction_page
    context["_render_3d_viewer_page"] = lambda: render_heart_viewer(context["HEART_VIEWER_DIR"])
    context["apply_custom_styles"] = apply_global_styles
    # The updated view also uses the corrected context renderer directly.
    globals()["_render_explainability_section"] = render_health_context
