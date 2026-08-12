import streamlit as st
import pandas as pd
import joblib

# =============================================================
# STEP 2: STREAMLIT APP
# Run locally with: streamlit run app.py
# Needs student_model.pkl, model_features.pkl, encoders.pkl
# in the SAME folder as this file (downloaded from Colab).
# =============================================================

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

# ---- Load model + supporting files ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("student_model.pkl")
    feature_order = joblib.load("model_features.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, feature_order, encoders

model, feature_order, encoders = load_artifacts()

st.title("🎓 Student Performance Predictor")
st.write("Fill in the student details below to predict whether they are likely to **pass or fail**.")

# ---- Numeric field ranges (from UCI dataset documentation) ----
numeric_ranges = {
    "age": (15, 22, 17),
    "Medu": (0, 4, 2),
    "Fedu": (0, 4, 2),
    "traveltime": (1, 4, 1),
    "studytime": (1, 4, 2),
    "failures": (0, 4, 0),
    "famrel": (1, 5, 4),
    "freetime": (1, 5, 3),
    "goout": (1, 5, 3),
    "Dalc": (1, 5, 1),
    "Walc": (1, 5, 1),
    "health": (1, 5, 3),
    "absences": (0, 93, 4),
}

field_help = {
    "Medu": "0=none, 1=primary, 2=5th-9th grade, 3=secondary, 4=higher education",
    "Fedu": "0=none, 1=primary, 2=5th-9th grade, 3=secondary, 4=higher education",
    "traveltime": "1=<15min, 2=15-30min, 3=30min-1hr, 4=>1hr",
    "studytime": "1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs (weekly)",
    "famrel": "Quality of family relationships (1=very bad, 5=excellent)",
    "Dalc": "Workday alcohol consumption (1=very low, 5=very high)",
    "Walc": "Weekend alcohol consumption (1=very low, 5=very high)",
    "health": "Current health status (1=very bad, 5=very good)",
}

# ---- Build form dynamically based on saved feature list ----
user_input = {}

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    columns_cycle = [col1, col2]

    for i, feature in enumerate(feature_order):
        target_col = columns_cycle[i % 2]

        if feature in encoders:
            # Categorical column -> dropdown with original labels
            options = list(encoders[feature].classes_)
            choice = target_col.selectbox(feature, options, help=field_help.get(feature))
            user_input[feature] = encoders[feature].transform([choice])[0]

        elif feature in numeric_ranges:
            lo, hi, default = numeric_ranges[feature]
            user_input[feature] = target_col.slider(
                feature, min_value=lo, max_value=hi, value=default,
                help=field_help.get(feature)
            )

        else:
            # Fallback for any unexpected numeric column
            user_input[feature] = target_col.number_input(feature, value=0)

    submitted = st.form_submit_button("Predict")

# ---- Predict ----
if submitted:
    input_df = pd.DataFrame([user_input])[feature_order]  # enforce correct column order
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Predicted: **PASS** (confidence: {proba[1]*100:.1f}%)")
    else:
        st.error(f"⚠️ Predicted: **AT RISK / FAIL** (confidence: {proba[0]*100:.1f}%)")

    st.caption(
        "Note: This prediction is based on the UCI Student Performance dataset "
        "and does not use the student's earlier-period grades (G1, G2), "
        "so it estimates risk from background/behavioral factors alone."
    )
