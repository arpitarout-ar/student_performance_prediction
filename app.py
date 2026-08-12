import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =============================================================
# STREAMLIT APP (v2) — Classification + Regression + Feature Importance
# Needs these files in the SAME folder:
#   student_model.pkl, model_features.pkl, encoders.pkl
#   student_model_regression.pkl, model_features_regression.pkl
# =============================================================

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

@st.cache_resource
def load_artifacts():
    clf_model = joblib.load("student_model.pkl")
    clf_features = joblib.load("model_features.pkl")
    encoders = joblib.load("encoders.pkl")

    reg_model, reg_features = None, None
    try:
        reg_model = joblib.load("student_model_regression.pkl")
        reg_features = joblib.load("model_features_regression.pkl")
    except FileNotFoundError:
        pass  # regression files are optional

    return clf_model, clf_features, encoders, reg_model, reg_features

clf_model, clf_features, encoders, reg_model, reg_features = load_artifacts()

st.title("🎓 Student Performance Predictor")

mode = st.radio(
    "Choose prediction type:",
    ["Pass / At-Risk (Classification)", "Predicted Final Grade (Regression)"],
    horizontal=True
)
use_regression = mode.startswith("Predicted")

if use_regression and reg_model is None:
    st.warning("Regression model files not found. Falling back to classification mode.")
    use_regression = False

feature_order = reg_features if use_regression else clf_features
active_model = reg_model if use_regression else clf_model

st.write("Fill in the student details below:")

# ---- Numeric field ranges ----
numeric_ranges = {
    "age": (15, 22, 17), "Medu": (0, 4, 2), "Fedu": (0, 4, 2),
    "traveltime": (1, 4, 1), "studytime": (1, 4, 2), "failures": (0, 4, 0),
    "famrel": (1, 5, 4), "freetime": (1, 5, 3), "goout": (1, 5, 3),
    "Dalc": (1, 5, 1), "Walc": (1, 5, 1), "health": (1, 5, 3), "absences": (0, 93, 4),
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

user_input = {}

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    columns_cycle = [col1, col2]

    for i, feature in enumerate(feature_order):
        target_col = columns_cycle[i % 2]

        if feature in encoders:
            options = list(encoders[feature].classes_)
            choice = target_col.selectbox(feature, options, help=field_help.get(feature), key=f"{feature}_{use_regression}")
            user_input[feature] = encoders[feature].transform([choice])[0]
        elif feature in numeric_ranges:
            lo, hi, default = numeric_ranges[feature]
            user_input[feature] = target_col.slider(
                feature, min_value=lo, max_value=hi, value=default,
                help=field_help.get(feature), key=f"{feature}_{use_regression}_slider"
            )
        else:
            user_input[feature] = target_col.number_input(feature, value=0, key=f"{feature}_{use_regression}_num")

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input])[feature_order]

    st.divider()
    if use_regression:
        pred_grade = active_model.predict(input_df)[0]
        st.success(f"📊 Predicted Final Grade: **{pred_grade:.1f} / 20**")
        status = "likely to pass" if pred_grade >= 10 else "at risk of failing"
        st.caption(f"A grade of {pred_grade:.1f} suggests this student is {status}.")
    else:
        prediction = active_model.predict(input_df)[0]
        proba = active_model.predict_proba(input_df)[0]
        if prediction == 1:
            st.success(f"✅ Predicted: **PASS** (confidence: {proba[1]*100:.1f}%)")
        else:
            st.error(f"⚠️ Predicted: **AT RISK / FAIL** (confidence: {proba[0]*100:.1f}%)")

    # ---- Feature Importance Explanation ----
    st.divider()
    st.subheader("🔍 Why this prediction? (Top factors the model relies on)")

    importances = pd.Series(active_model.feature_importances_, index=feature_order)
    top_features = importances.sort_values(ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(6, 4))
    top_features.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title("Top factors driving predictions (overall model, not just this student)")
    st.pyplot(fig)

    st.caption(
        "This chart shows which features the model relies on most heavily across "
        "all predictions — not specifically for this student's inputs. It gives a "
        "general sense of what matters most (e.g. absences, past failures, study time)."
    )

st.divider()
st.caption(
    "Note: Predictions do not use the student's earlier-period grades (G1, G2), "
    "so they estimate outcomes from background/behavioral factors alone — "
    "making this useful for early intervention, before those grades exist."
)
