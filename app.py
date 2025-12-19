import streamlit as st
import pandas as pd
import joblib
import os
import csv

# Load trained model
model = joblib.load("rf_model.joblib")

# Load dataset to align columns
df = pd.read_csv("churn_cleaned.csv")
X = df.drop("Churn", axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)

# Feedback file
FEEDBACK_FILE = "user_feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Role", "Rating", "Useful", "Suggestions"])

# Title
st.markdown("<h1 style='text-align: center; color: teal;'>Telecom Churn Prediction</h1>", unsafe_allow_html=True)

st.markdown("---")

# Input fields
st.subheader("Enter Customer Details:")
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tech = st.selectbox("Tech Support", ["Yes", "No"])
security = st.selectbox("Online Security", ["Yes", "No"])

# Submit button
if st.button("Predict Churn Probability"):
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "TechSupport": tech,
        "OnlineSecurity": security
    }

    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    prob = model.predict_proba(input_encoded)[0][1]
    input_data["Probability"] = round(prob, 2)

    # Store in session state
    if "submissions" not in st.session_state:
        st.session_state.submissions = []
    st.session_state.submissions.append(input_data)

    st.success(f"Churn Probability: {round(prob, 2)}")

# Display all previous submissions
if "submissions" in st.session_state and st.session_state.submissions:
    st.subheader("Submitted Entries:")
    st.table(pd.DataFrame(st.session_state.submissions))

st.markdown("---")

# Feedback section
st.subheader("User Feedback")
with st.form("feedback_form"):
    name = st.text_input("Name")
    role = st.text_input("Role")
    rating = st.slider("Rating (1-5)", min_value=1, max_value=5)
    useful = st.selectbox("Was this useful?", ["Yes", "No"])
    suggestions = st.text_area("Suggestions")

    submitted = st.form_submit_button("Submit Feedback")
    if submitted:
        with open(FEEDBACK_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, role, rating, useful, suggestions])
        st.success("Thank you for your feedback!")
