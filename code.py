from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import sys

# -------------------------------
# Load dataset safely
# -------------------------------
try:
    df = pd.read_csv("study_time_vs_marks_dataset.csv")
except FileNotFoundError:
    print("Error: Dataset file 'study_time_vs_marks_dataset.csv' not found.")
    sys.exit()

# Remove missing values
df = df.dropna()

# -------------------------------
# Features and target
# -------------------------------
x = df[["study_hours", "sleep_hours"]]
y = df["exam_score"]

# -------------------------------
# Train model
# -------------------------------
def train_model():
    model = LinearRegression()
    model.fit(x, y)
    return model

model = train_model()

# -------------------------------
# Prediction function
# -------------------------------
def predict_score(study_hours: float, sleep_hours: float) -> float:
    input_df = pd.DataFrame(
        [[study_hours, sleep_hours]],
        columns=["study_hours", "sleep_hours"]
    )
    return float(model.predict(input_df)[0])

# -------------------------------
# Try Streamlit UI
# -------------------------------
try:
    import streamlit as st

    # Cache model (for Streamlit performance)
    @st.cache_resource
    def get_model():
        return train_model()

    model = get_model()

    st.title("Study Hours vs Exam Score Predictor")

    st.markdown("Use sliders or choose a preset:")

    study_hours = st.slider("Study hours", 0.0, 12.0, 4.0, 0.5)
    sleep_hours = st.slider("Sleep hours", 0.0, 12.0, 7.0, 0.5)

    preset = st.selectbox(
        "Preset options",
        [
            "Custom",
            "Low study / low sleep",
            "High study / good sleep",
            "Moderate study / moderate sleep"
        ]
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
    st.write({
        "study_hours": study_hours,
        "sleep_hours": sleep_hours
    })

# -------------------------------
# Console fallback
# -------------------------------
except ImportError:
    print("Running in console mode...\n")

    try:
        user_input1 = float(input("Enter study hours: "))
        user_input2 = float(input("Enter sleep hours: "))
    except ValueError:
        print("Error: Please enter valid numeric values.")
        sys.exit()

    predicted_score = predict_score(user_input1, user_input2)
    print(f"Predicted exam score: {predicted_score:.2f}")
