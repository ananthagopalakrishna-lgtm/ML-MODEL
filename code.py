from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import sys

# Load dataset
df = pd.read_csv("study_time_vs_marks_dataset.csv")

# Features and target
x = df[["study_hours", "sleep_hours"]]
y = df["exam_score"]

# Train model
model = LinearRegression()
model.fit(x, y)

# Prediction helper
def predict_score(study_hours: float, sleep_hours: float) -> float:
    input_df = pd.DataFrame([[study_hours, sleep_hours]],
                            columns=["study_hours", "sleep_hours"])
    return float(model.predict(input_df)[0])

# Streamlit UI (if available)
try:
    import streamlit as st
except ImportError:
    st = None

streamlit_active = False
if st is not None:
    try:
        streamlit_active = bool(getattr(st, '_is_running_with_streamlit', False))
    except Exception:
        streamlit_active = False

if streamlit_active:
    st.title("Study Hours vs Exam Score Predictor")

    st.markdown("Use sliders or the preset dropdown for inputs.")

    study_hours = st.slider("Study hours", 0.0, 12.0, 4.0, 0.5)
    sleep_hours = st.slider("Sleep hours", 0.0, 12.0, 7.0, 0.5)

    preset = st.selectbox(
        "Or select a preset",
        ["Custom", "Low study / low sleep", "High study / good sleep", "Moderate study / moderate sleep"]
    )

    if preset == "Low study / low sleep":
        study_hours, sleep_hours = 2.0, 5.0
    elif preset == "High study / good sleep":
        study_hours, sleep_hours = 10.0, 8.0
    elif preset == "Moderate study / moderate sleep":
        study_hours, sleep_hours = 6.0, 7.0

    if st.button("Predict"):
        pred = predict_score(study_hours, sleep_hours)
        st.metric("Predicted exam score", f"{pred:.2f}")

    st.write("### Current inputs")
    st.write({"study_hours": study_hours, "sleep_hours": sleep_hours})

else:
    # Console fallback (existing behavior preserved)
    user_input1 = float(input("Enter study hours: "))
    user_input2 = float(input("Enter sleep hours: "))

    predicted_score = predict_score(user_input1, user_input2)
    print(f"Predicted exam score: {predicted_score:.2f}")
