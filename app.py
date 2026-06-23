import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import os

# Train model if not already trained
@st.cache_resource
def load_model():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop('customerID', axis=1)

    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1,'No':0,'Male':1,'Female':0})

    df = pd.get_dummies(df, columns=['MultipleLines','InternetService','OnlineSecurity',
                                      'OnlineBackup','DeviceProtection','TechSupport',
                                      'StreamingTV','StreamingMovies','Contract','PaymentMethod'])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    scale_cols = ['tenure','MonthlyCharges','TotalCharges']
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3,
                          random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    return model, scaler, list(X.columns)

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details to predict if they will churn.")

with st.spinner("Loading model..."):
    model, scaler, feature_cols = load_model()

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

input_dict = {
    'gender': 1 if gender=='Male' else 0,
    'SeniorCitizen': 1 if senior=='Yes' else 0,
    'Partner': 1 if partner=='Yes' else 0,
    'Dependents': 1 if dependents=='Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service=='Yes' else 0,
    'PaperlessBilling': 1 if paperless_billing=='Yes' else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'MultipleLines_No': 1 if multiple_lines=='No' else 0,
    'MultipleLines_No phone service': 1 if multiple_lines=='No phone service' else 0,
    'MultipleLines_Yes': 1 if multiple_lines=='Yes' else 0,
    'InternetService_DSL': 1 if internet_service=='DSL' else 0,
    'InternetService_Fiber optic': 1 if internet_service=='Fiber optic' else 0,
    'InternetService_No': 1 if internet_service=='No' else 0,
    'OnlineSecurity_No': 1 if online_security=='No' else 0,
    'OnlineSecurity_No internet service': 1 if online_security=='No internet service' else 0,
    'OnlineSecurity_Yes': 1 if online_security=='Yes' else 0,
    'OnlineBackup_No': 1 if online_backup=='No' else 0,
    'OnlineBackup_No internet service': 1 if online_backup=='No internet service' else 0,
    'OnlineBackup_Yes': 1 if online_backup=='Yes' else 0,
    'DeviceProtection_No': 1 if device_protection=='No' else 0,
    'DeviceProtection_No internet service': 1 if device_protection=='No internet service' else 0,
    'DeviceProtection_Yes': 1 if device_protection=='Yes' else 0,
    'TechSupport_No': 1 if tech_support=='No' else 0,
    'TechSupport_No internet service': 1 if tech_support=='No internet service' else 0,
    'TechSupport_Yes': 1 if tech_support=='Yes' else 0,
    'StreamingTV_No': 1 if streaming_tv=='No' else 0,
    'StreamingTV_No internet service': 1 if streaming_tv=='No internet service' else 0,
    'StreamingTV_Yes': 1 if streaming_tv=='Yes' else 0,
    'StreamingMovies_No': 1 if streaming_movies=='No' else 0,
    'StreamingMovies_No internet service': 1 if streaming_movies=='No internet service' else 0,
    'StreamingMovies_Yes': 1 if streaming_movies=='Yes' else 0,
    'Contract_Month-to-month': 1 if contract=='Month-to-month' else 0,
    'Contract_One year': 1 if contract=='One year' else 0,
    'Contract_Two year': 1 if contract=='Two year' else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if payment_method=='Bank transfer (automatic)' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method=='Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if payment_method=='Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment_method=='Mailed check' else 0,
}

input_df = pd.DataFrame([input_dict])
input_df[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
    input_df[['tenure','MonthlyCharges','TotalCharges']])

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ This customer is **LIKELY TO CHURN**")
    else:
        st.success("✅ This customer is **NOT LIKELY TO CHURN**")

    st.metric("Churn Probability", f"{probability*100:.1f}%")

    st.markdown("---")
    st.subheader("Key Risk Factors")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tenure", f"{tenure} months")
    col2.metric("Monthly Charges", f"${monthly_charges}")
    col3.metric("Contract", contract)