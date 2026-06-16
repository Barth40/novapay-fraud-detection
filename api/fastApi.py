from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("models/random_forest_model.pkl")
model_features = joblib.load("models/model_features.pkl")

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    amount_usd: float = 5000.0
    fee: float = 3.5
    exchange_rate_src_to_dest: float = 1.0
    new_device: int = 1
    location_mismatch: int = 1
    ip_risk_score: float = 0.3
    account_age_days: int = 300
    device_trust_score: float = 0.1
    chargeback_history_count: int = 0
    risk_score_internal: float = 0.85
    txn_velocity_1h: int = 5
    txn_velocity_24h: int = 8
    corridor_risk: float = 0.2
    hour: int = 12
    month: int = 6

    channel: str = "web"
    kyc_tier: str = "low"
    source_currency: str = "usd"
    dest_currency: str = "cad"
    ip_country: str = "us"
    home_country: str = "us"
    day_of_week: str = "Monday"


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    amount_usd = transaction.amount_usd
    fee = transaction.fee
    exchange_rate_src_to_dest = transaction.exchange_rate_src_to_dest
    new_device = transaction.new_device
    location_mismatch = transaction.location_mismatch
    ip_risk_score = transaction.ip_risk_score
    account_age_days = transaction.account_age_days
    device_trust_score = transaction.device_trust_score
    chargeback_history_count = transaction.chargeback_history_count
    risk_score_internal = transaction.risk_score_internal
    txn_velocity_1h = transaction.txn_velocity_1h
    txn_velocity_24h = transaction.txn_velocity_24h
    corridor_risk = transaction.corridor_risk
    hour = transaction.hour
    month = transaction.month

    channel = transaction.channel
    kyc_tier = transaction.kyc_tier
    source_currency = transaction.source_currency
    dest_currency = transaction.dest_currency
    ip_country = transaction.ip_country
    home_country = transaction.home_country
    day_of_week = transaction.day_of_week

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

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        result = "Fraudulent Transaction Detected"
    else:
        result = "Transaction Appears Legitimate"

    return {
        "transaction_input": input_data.to_dict(orient="records")[0],
        "prediction": int(prediction),
        "result": result,
        "fraud_probability": f"{probability:.2%}"
    }