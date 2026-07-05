import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Customer Churn Prediction App")

# Load trained model
pipe = pickle.load(open("pipe.pkl", "rb"))

# Load dataset (for dropdown options)
df = pd.read_csv("final_data.csv")

# Sidebar inputs
customerID = st.sidebar.text_input("Enter Customer ID")

gender = st.sidebar.selectbox("Gender", df["gender"].unique())
SeniorCitizen = st.sidebar.selectbox("Senior Citizen (0/1)", [0, 1])
Partner = st.sidebar.selectbox("Partner", df["Partner"].unique())
Dependents = st.sidebar.selectbox("Dependents", df["Dependents"].unique())
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
PhoneService = st.sidebar.selectbox("Phone Service", df["PhoneService"].unique())
MultipleLines = st.sidebar.selectbox("Multiple Lines", df["MultipleLines"].unique())
InternetService = st.sidebar.selectbox("Internet Service", df["InternetService"].unique())
OnlineSecurity = st.sidebar.selectbox("Online Security", df["OnlineSecurity"].unique())
OnlineBackup = st.sidebar.selectbox("Online Backup", df["OnlineBackup"].unique())
DeviceProtection = st.sidebar.selectbox("Device Protection", df["DeviceProtection"].unique())
TechSupport = st.sidebar.selectbox("Tech Support", df["TechSupport"].unique())
StreamingTV = st.sidebar.selectbox("Streaming TV", df["StreamingTV"].unique())
StreamingMovies = st.sidebar.selectbox("Streaming Movies", df["StreamingMovies"].unique())
Contract = st.sidebar.selectbox("Contract", df["Contract"].unique())
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", df["PaperlessBilling"].unique())
PaymentMethod = st.sidebar.selectbox("Payment Method", df["PaymentMethod"].unique())
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0)

# Predict button
if st.sidebar.button("Predict Churn"):
    
    st.write("### Selected Customer Details")
    st.write(f"Gender: {gender}")
    st.write(f"SeniorCitizen: {SeniorCitizen}")
    st.write(f"Partner: {Partner}")
    st.write(f"Dependents: {Dependents}")
    st.write(f"Tenure: {tenure}")
    st.write(f"PhoneService: {PhoneService}")
    st.write(f"InternetService: {InternetService}")
    st.write(f"Contract: {Contract}")
    st.write(f"MonthlyCharges: {MonthlyCharges}")
    st.write(f"TotalCharges: {TotalCharges}")

    # Input DataFrame
    myinput = [[
        gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges
    ]]

    columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

    myinput = pd.DataFrame(myinput, columns=columns)

    # Prediction
    result = pipe.predict(myinput)

    if result[0] == 1:
        st.error("Customer will CHURN (Service will be discontinued)")
    else:
        st.success(" Customer will NOT CHURN (Customer will stay)")