import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("models/random_forest_model.pkl")
model_features = joblib.load("models/model_features.pkl")

st.set_page_config(
    page_title="Fraud Detection App",
    page_icon=" ",
    layout="wide"
)

st.title("💳 Fraud Detection ML App")
st.write("This app predicts whether a transaction is likely to be fraudulent.")

st.sidebar.header("Transaction Details")

amount_usd = st.sidebar.number_input("Amount in USD", min_value=0.0, value=5000.0)
fee = st.sidebar.number_input("Fee", min_value=0.0, value=3.5)
exchange_rate_src_to_dest = st.sidebar.number_input("Exchange Rate", min_value=0.0, value=1.0)
new_device = st.sidebar.selectbox("New Device?", [0, 1])
location_mismatch = st.sidebar.selectbox("Location Mismatch?", [0, 1])
ip_risk_score = st.sidebar.slider("IP Risk Score", 0.0, 1.0, 0.3)
account_age_days = st.sidebar.number_input("Account Age Days", min_value=0, value=300)
device_trust_score = st.sidebar.slider("Device Trust Score", 0.0, 1.0, 0.1)
chargeback_history_count = st.sidebar.number_input("Chargeback History Count", min_value=0, value=0)
risk_score_internal = st.sidebar.slider("Internal Risk Score", 0.0, 1.0, 0.85)
txn_velocity_1h = st.sidebar.number_input("Transaction Velocity 1h", min_value=0, value=5)
txn_velocity_24h = st.sidebar.number_input("Transaction Velocity 24h", min_value=0, value=8)
corridor_risk = st.sidebar.slider("Corridor Risk", 0.0, 1.0, 0.2)
hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)
month = st.sidebar.slider("Transaction Month", 1, 12, 6)

channel = st.sidebar.selectbox("Channel", ["mobile", "web", "atm", "unknown"])
kyc_tier = st.sidebar.selectbox("KYC Tier", ["standard", "enhanced", "low", "unknown"])
source_currency = st.sidebar.selectbox("Source Currency", ["usd", "gbp", "cad"])
dest_currency = st.sidebar.selectbox("Destination Currency", ["cad", "mxn", "cny", "eur", "inr"])
ip_country = st.sidebar.selectbox("IP Country", ["us", "uk", "ca", "unknown"])
home_country = st.sidebar.selectbox("Home Country", ["us", "uk", "ca"])
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

amount_src = amount_usd
amount_dest = amount_src * exchange_rate_src_to_dest

# Create input dataframe
input_data = pd.DataFrame({
    "amount_src": [amount_src],
    "amount_usd": [amount_usd],
    "fee": [fee],
    "exchange_rate_src_to_dest": [exchange_rate_src_to_dest],
    "new_device": [new_device],
    "location_mismatch": [location_mismatch],
    "ip_risk_score": [ip_risk_score],
    "account_age_days": [account_age_days],
    "device_trust_score": [device_trust_score],
    "chargeback_history_count": [chargeback_history_count],
    "risk_score_internal": [risk_score_internal],
    "txn_velocity_1h": [txn_velocity_1h],
    "txn_velocity_24h": [txn_velocity_24h],
    "corridor_risk": [corridor_risk],
    "hour": [hour],
    "month": [month],
    "amount_dest": [amount_dest],
    "channel": [channel],
    "kyc_tier": [kyc_tier],
    "source_currency": [source_currency],
    "dest_currency": [dest_currency],
    "ip_country": [ip_country],
    "home_country": [home_country],
    "day_of_week": [day_of_week]
})

# One-hot encode input
input_encoded = pd.get_dummies(input_data)

# Align with training columns
input_encoded = input_encoded.reindex(
    columns=model_features,
    fill_value=0
)

st.subheader("Transaction Input")
st.dataframe(input_data)

if st.button("Predict Fraud Risk"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Fraudulent Transaction Detected")
        st.write(f"Fraud Probability: **{probability:.2%}**")
    else:
        st.success("Transaction Appears Legitimate")
        st.write(f"Fraud Probability: **{probability:.2%}**")