import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

# Title
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details to predict if they will churn.")

# Sidebar inputs
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)

# Build input dataframe
input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'MultipleLines_No': 1 if multiple_lines == 'No' else 0,
    'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
    'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
    'InternetService_DSL': 1 if internet_service == 'DSL' else 0,
    'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
    'InternetService_No': 1 if internet_service == 'No' else 0,
    'OnlineSecurity_No': 1 if online_security == 'No' else 0,
    'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
    'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
    'OnlineBackup_No': 1 if online_backup == 'No' else 0,
    'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
    'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
    'DeviceProtection_No': 1 if device_protection == 'No' else 0,
    'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
    'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
    'TechSupport_No': 1 if tech_support == 'No' else 0,
    'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
    'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
    'StreamingTV_No': 1 if streaming_tv == 'No' else 0,
    'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
    'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
    'StreamingMovies_No': 1 if streaming_movies == 'No' else 0,
    'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
    'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
    'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == 'Bank transfer (automatic)' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
}

input_df = pd.DataFrame([input_dict])

# Scale numeric columns
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Predict button
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ This customer is **LIKELY TO CHURN**")
    else:
        st.success(f"✅ This customer is **NOT LIKELY TO CHURN**")

    st.metric("Churn Probability", f"{probability*100:.1f}%")

    st.markdown("---")
    st.subheader("Key Risk Factors")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tenure", f"{tenure} months")
    col2.metric("Monthly Charges", f"${monthly_charges}")
    col3.metric("Contract", contract)