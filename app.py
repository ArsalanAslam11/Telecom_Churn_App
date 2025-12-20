import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ---------------------------
# Load Model Pipeline
# ---------------------------
MODEL_FILE = "rf_model.joblib"

if not os.path.exists(MODEL_FILE):
    st.error("Model file not found. Please train the model first using train_model.py")
    st.stop()

model = joblib.load(MODEL_FILE)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Telecom Churn Prediction", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>Telecom Churn Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#555;'>Enter customer details to predict if they will churn</p>",
    unsafe_allow_html=True
)

# ---------------------------
# Input Form
# ---------------------------
with st.form("prediction_form"):
    st.markdown("#### Customer Information")
    
    col1, col2 = st.columns(2)
    with col1:
        customer_id = st.text_input("Customer ID (optional)")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.1)
    
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    submit = st.form_submit_button("Predict Churn", use_container_width=True)

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

    if prediction == 1:
        st.error("High Risk of Churn")
    else:
        st.success("Low Risk of Churn")

    st.info(f"Churn Probability: {probability:.2%}")

    record = {
        "CustomerID": customer_id or "N/A",
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "Prediction": result,
        "Probability": f"{probability:.2%}",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    file = "predictions.csv"
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(file, index=False)
    st.success("Prediction saved to history")

# ---------------------------
# Prediction History + Owner Delete Tools
# ---------------------------
if os.path.exists("predictions.csv"):
    st.markdown("### Prediction History")
    history_df = pd.read_csv("predictions.csv")
    
    if "Timestamp" in history_df.columns:
        history_df = history_df.sort_values("Timestamp", ascending=False)
    else:
        history_df = history_df.sort_index(ascending=False)
    
    history_df = history_df.reset_index(drop=True)
    st.dataframe(history_df, use_container_width=True)

    st.markdown("#### Owner Tools (Password Protected)")
    OWNER_PASSWORD = "225191arsalan8090"  

    with st.expander("Delete All Prediction Records (Owner Only)"):
        password_all = st.text_input("Enter owner password to delete all records", type="password", key="pwd_pred_all")
        col_del, _ = st.columns([1, 3])
        with col_del:
            delete_all_btn = st.button("Delete All Records", key="del_all_pred")

        if delete_all_btn:
            if password_all == OWNER_PASSWORD:
                os.remove("predictions.csv")
                st.success("All prediction records have been deleted.")
                st.rerun()
            else:
                st.error("Incorrect password. Access denied.")

    with st.expander("Delete Specific Prediction Record (Owner Only)"):
        if len(history_df) > 0:
            password_one = st.text_input("Enter owner password", type="password", key="pwd_pred_one")

            row_to_delete = st.selectbox(
                "Select row to delete (0 = newest)",
                options=list(range(len(history_df))),
                format_func=lambda x: f"Row {x}: {history_df.iloc[x]['CustomerID']} - {history_df.iloc[x]['Prediction']}"
            )

            col_one, _ = st.columns([1, 3])
            with col_one:
                delete_one_btn = st.button("Delete Selected Record", key="del_one_pred")

            if delete_one_btn:
                if password_one == OWNER_PASSWORD:
                    history_df = history_df.drop(history_df.index[row_to_delete]).reset_index(drop=True)
                    history_df.to_csv("predictions.csv", index=False)
                    st.success("Selected record has been deleted.")
                    st.rerun()
                else:
                    st.error("Incorrect password. Access denied.")
        else:
            st.info("No records available to delete.")

else:
    st.info("No prediction history yet.")

# ---------------------------
# Feedback Section
# ---------------------------
st.markdown("---")
st.markdown("### Submit Feedback")

with st.form("feedback_form"):
    name = st.text_input("Your Name *", placeholder="e.g., Arsalan")
    feedback = st.text_area("Your Feedback *", placeholder="What do you think about this app? Any suggestions?")

    fb_submit = st.form_submit_button("Submit Feedback", use_container_width=True)

if fb_submit:
    if not name.strip():
        st.warning("Please enter your name.")
    elif not feedback.strip():
        st.warning("Please write some feedback.")
    else:
        fb_record = {
            "Name": name.strip(),
            "Feedback": feedback.strip(),
            "Submitted_On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        fb_file = "user_feedback.csv"
        if os.path.exists(fb_file):
            df_fb = pd.read_csv(fb_file)
            if "Submitted_On" in df_fb.columns:
                df_fb = df_fb[["Name", "Feedback", "Submitted_On"]]
            else:
                df_fb = df_fb[["Name", "Feedback"]]
            df_fb = pd.concat([df_fb, pd.DataFrame([fb_record])], ignore_index=True)
        else:
            df_fb = pd.DataFrame([fb_record])

        df_fb.to_csv(fb_file, index=False)
        st.success("Thank you. Your feedback has been recorded.")

# ---------------------------
# Previous Feedback + Owner Delete Tools
# ---------------------------
if os.path.exists("user_feedback.csv"):
    st.markdown("### Previous Feedback")
    try:
        feedback_df = pd.read_csv("user_feedback.csv")
        display_columns = ["Submitted_On", "Name", "Feedback"] if "Submitted_On" in feedback_df.columns else ["Name", "Feedback"]
        if display_columns[0] in feedback_df.columns:
            feedback_df = feedback_df.sort_values(display_columns[0], ascending=False)
        feedback_df = feedback_df.reset_index(drop=True)
        st.dataframe(feedback_df[display_columns], use_container_width=True)

        # Owner Tools for Feedback
        st.markdown("#### Owner Tools - Feedback Management (Password Protected)")
        OWNER_PASSWORD = "225191arsalan8090"  

        with st.expander("Delete All Feedback (Owner Only)"):
            pwd_fb_all = st.text_input("Enter owner password to delete all feedback", type="password", key="pwd_fb_all")
            c1, _ = st.columns([1, 3])
            with c1:
                del_all_fb = st.button("Delete All Feedback", key="del_all_fb")

            if del_all_fb:
                if pwd_fb_all == OWNER_PASSWORD:
                    os.remove("user_feedback.csv")
                    st.success("All feedback has been deleted.")
                    st.rerun()
                else:
                    st.error("Incorrect password. Access denied.")

        with st.expander("Delete Specific Feedback (Owner Only)"):
            if len(feedback_df) > 0:
                pwd_fb_one = st.text_input("Enter owner password", type="password", key="pwd_fb_one")

                fb_row = st.selectbox(
                    "Select feedback to delete (0 = newest)",
                    options=list(range(len(feedback_df))),
                    format_func=lambda x: f"Row {x}: {feedback_df.iloc[x]['Name']} - \"{feedback_df.iloc[x]['Feedback'][:50]}{'...' if len(feedback_df.iloc[x]['Feedback']) > 50 else ''}\""
                )

                c2, _ = st.columns([1, 3])
                with c2:
                    del_one_fb = st.button("Delete Selected Feedback", key="del_one_fb")

                if del_one_fb:
                    if pwd_fb_one == OWNER_PASSWORD:
                        feedback_df = feedback_df.drop(feedback_df.index[fb_row]).reset_index(drop=True)
                        feedback_df[display_columns].to_csv("user_feedback.csv", index=False)
                        st.success("Selected feedback has been deleted.")
                        st.rerun()
                    else:
                        st.error("Incorrect password. Access denied.")
            else:
                st.info("No feedback available to delete.")

    except Exception:
        st.error("Error loading feedback data. You may delete 'user_feedback.csv' to reset.")
else:
    st.info("No feedback submitted yet.")