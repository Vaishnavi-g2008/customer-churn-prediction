import streamlit as st
import pandas as pd
import pickle

st.title("Customer Churn Prediction App")

# Load model
pipe = pickle.load(open("pipe.pkl", "rb"))

# Load dataset
df = pd.read_csv("final_data.csv")

# Sidebar Inputs
gender = st.sidebar.selectbox("Gender", sorted(df["gender"].unique()))
partner = st.sidebar.selectbox("Partner", sorted(df["Partner"].unique()))
dependents = st.sidebar.selectbox("Dependents", sorted(df["Dependents"].unique()))
phone_service = st.sidebar.selectbox("Phone Service", sorted(df["PhoneService"].unique()))
multiple_lines = st.sidebar.selectbox("Multiple Lines", sorted(df["MultipleLines"].unique()))
internet_service = st.sidebar.selectbox("Internet Service", sorted(df["InternetService"].unique()))
online_security = st.sidebar.selectbox("Online Security", sorted(df["OnlineSecurity"].unique()))
online_backup = st.sidebar.selectbox("Online Backup", sorted(df["OnlineBackup"].unique()))
device_protection = st.sidebar.selectbox("Device Protection", sorted(df["DeviceProtection"].unique()))
tech_support = st.sidebar.selectbox("Tech Support", sorted(df["TechSupport"].unique()))
streaming_tv = st.sidebar.selectbox("Streaming TV", sorted(df["StreamingTV"].unique()))
streaming_movies = st.sidebar.selectbox("Streaming Movies", sorted(df["StreamingMovies"].unique()))
contract = st.sidebar.selectbox("Contract", sorted(df["Contract"].unique()))
paperless_billing = st.sidebar.selectbox("Paperless Billing", sorted(df["PaperlessBilling"].unique()))
payment_method = st.sidebar.selectbox("Payment Method", sorted(df["PaymentMethod"].unique()))

if st.sidebar.button("Predict"):

    myinput = pd.DataFrame(
        [[
            gender,
            partner,
            dependents,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method
        ]],
        columns=[
            'gender',
            'Partner',
            'Dependents',
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod'
        ]
    )

    result = pipe.predict(myinput)

    st.subheader("Prediction Result")

    if result[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is Not Likely to Churn")