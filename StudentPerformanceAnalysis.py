import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

st.title("🎓 Student Performance Prediction App")

# -------------------------------
# Load Saved Model & Preprocessing
# -------------------------------
model = joblib.load("Student_model.pkl")
scaler = joblib.load("Performance_scaler.pkl")
selector = joblib.load("selected_features.pkl")
raw_columns = joblib.load("Input_values.pkl")
x_columns = [str(col) for col in raw_columns]

st.sidebar.header("Enter Student Details")

# -------------------------------
# Create Sidebar Inputs Dynamically
# -------------------------------
input_values = {}

for col in x_columns:
    input_values[col] = st.sidebar.number_input(
        f"{col}",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=1.0
    )

# Convert to DataFrame
input_df = pd.DataFrame([input_values], columns=x_columns)

# -------------------------------
# Select Best Features → Scale → Predict
# -------------------------------
try:
    # Apply SelectKBest
    selected_features = selector.transform(input_df)

    # Scale input
    scaled_features = scaler.transform(selected_features)

    # Prediction
    prediction = model.predict(scaled_features)[0]

    st.subheader("📘 Prediction Result")
    st.success(f"Predicted Score: **{prediction:.2f}** / 100")

    # Appreciation message
    if prediction >= 90:
        st.info("🌟 **Excellent performance! Keep it up!**")
    elif prediction >= 75:
        st.info("👍 **Very good! You can achieve even more!**")
    elif prediction >= 50:
        st.info("🙂 **Average performance. A little more effort can boost it!**")
    else:
        st.warning("📉 Needs improvement. Try increasing study time and consistency.")

except Exception as e:
    st.error(f"Error during prediction: {e}")

if st.button("Predict"):
    prediction = model.predict(selected_data)[0]
    st.success(f"Predicted Performance: {prediction:.2f}")
    if(prediction>=95):
        feedback="Your Performance was very Excellent"
    elif(prediction>=75 and prediction<95):
        feedback="Your Performance was Good"
    elif(prediction>=50 and prediction<75):
        feedback="Your Preformance was Average need some Improvement"
    else:
        feedback="Got Fail need to make Practice"

    st.success(feedback)