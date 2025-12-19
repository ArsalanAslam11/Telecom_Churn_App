import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------
# Load your trained model
# ---------------------------
MODEL_FILE = "rf_model.joblib"
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error("Model file not found!")
    st.stop()

# ---------------------------
# Page title
# ---------------------------
st.set_page_config(page_title="📊 Telecom Churn Prediction", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 Telecom Churn Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center; color: #333;'>Enter customer details below and predict churn probability.</p>", unsafe_allow_html=True)

# ---------------------------
# Input form
# ---------------------------
with st.form(key="prediction_form"):
    customer_id = st.text_input("Customer ID")
    tenure = st.number_input("Tenure (Months)", min_value=0)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    
    submit_button = st.form_submit_button("Predict")

# ---------------------------
# Handle prediction
# ---------------------------
if submit_button:
    # Encode contract type if your model requires numeric encoding
    contract_mapping = {"Month-to-Month":0, "One Year":1, "Two Year":2}
    contract_encoded = contract_mapping[contract_type]

    # Prepare input (update columns based on your model)
    input_data = [[tenure, monthly_charges, contract_encoded]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction for Customer {customer_id}: {prediction}")

    # Save input + prediction to a CSV
    record = {"CustomerID": customer_id, "Tenure": tenure, "MonthlyCharges": monthly_charges,
              "ContractType": contract_type, "Prediction": prediction}
    
    # Check if CSV exists
    predictions_file = "predictions.csv"
    if os.path.exists(predictions_file):
        df = pd.read_csv(predictions_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(predictions_file, index=False)

# ---------------------------
# Show prediction history
# ---------------------------
if os.path.exists("predictions.csv"):
    st.markdown("### 📝 Prediction History")
    st.table(pd.read_csv("predictions.csv"))

# ---------------------------
# Feedback section
# ---------------------------
st.markdown("### 💬 Submit Feedback")
with st.form(key="feedback_form"):
    feedback_name = st.text_input("Your Name")
    feedback_text = st.text_area("Your Feedback")
    feedback_button = st.form_submit_button("Submit Feedback")

if feedback_button:
    feedback_record = {"Name": feedback_name, "Feedback": feedback_text}

    feedback_file = "user_feedback.csv"
    if os.path.exists(feedback_file):
        df_feedback = pd.read_csv(feedback_file)
        df_feedback = pd.concat([df_feedback, pd.DataFrame([feedback_record])], ignore_index=True)
    else:
        df_feedback = pd.DataFrame([feedback_record])
    df_feedback.to_csv(feedback_file, index=False)

    st.success("✅ Thank you! Your feedback has been recorded.")

# ---------------------------
# Show previous feedback
# ---------------------------
if os.path.exists("user_feedback.csv"):
    st.markdown("### 🗣️ Previous Feedback")
    st.table(pd.read_csv("user_feedback.csv"))
