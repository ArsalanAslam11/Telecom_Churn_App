import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------
# Load Model Pipeline
# ---------------------------
MODEL_FILE = "rf_model.joblib"

if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found. Train the model first.")
    st.stop()

model = joblib.load(MODEL_FILE)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="📊 Telecom Churn Prediction", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>📊 Telecom Churn Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Enter customer details to predict churn</p>",
    unsafe_allow_html=True
)

# ---------------------------
# Input Form
# ---------------------------
with st.form("prediction_form"):
    customer_id = st.text_input("Customer ID")
    tenure = st.number_input("Tenure (Months)", min_value=0)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    submit = st.form_submit_button("Predict")

# ---------------------------
# Prediction
# ---------------------------
if submit:
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"

    st.success(f"Prediction: **{result}**")
    st.info(f"Churn Probability: **{probability:.2%}**")

    # Save history
    record = {
        "CustomerID": customer_id,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "Prediction": result,
        "Probability": probability
    }

    file = "predictions.csv"
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(file, index=False)

# ---------------------------
# Prediction History
# ---------------------------
if os.path.exists("predictions.csv"):
    st.markdown("### 📝 Prediction History")
    st.dataframe(pd.read_csv("predictions.csv"))

# ---------------------------
# Feedback
# ---------------------------
st.markdown("### 💬 Feedback")

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    feedback = st.text_area("Your Feedback")
    fb_submit = st.form_submit_button("Submit")

if fb_submit:
    fb_record = {"Name": name, "Feedback": feedback}
    fb_file = "user_feedback.csv"

    if os.path.exists(fb_file):
        df_fb = pd.read_csv(fb_file)
        df_fb = pd.concat([df_fb, pd.DataFrame([fb_record])], ignore_index=True)
    else:
        df_fb = pd.DataFrame([fb_record])

    df_fb.to_csv(fb_file, index=False)
    st.success("✅ Feedback saved")

if os.path.exists("user_feedback.csv"):
    st.markdown("### 🗣 Previous Feedback")
    st.dataframe(pd.read_csv("user_feedback.csv"))
