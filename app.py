
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from predict import predict_customer

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("End-to-end machine learning demo built for a Data Science & ML internship project.")

st.write("Enter customer details to estimate the probability of churn.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    payment = st.selectbox("Payment method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    monthly = st.number_input("Monthly charges", 18.0, 125.0, 70.0, step=1.0)
    total = st.number_input("Total charges", 0.0, 10000.0, 900.0, step=50.0)

customer = {
    "gender": gender,
    "senior_citizen": senior,
    "partner": partner,
    "dependents": dependents,
    "tenure_months": tenure,
    "phone_service": phone,
    "internet_service": internet,
    "contract": contract,
    "payment_method": payment,
    "tech_support": support,
    "streaming_tv": tv,
    "streaming_movies": movies,
    "paperless_billing": paperless,
    "monthly_charges": monthly,
    "total_charges": total,
}

if st.button("Predict churn", type="primary", use_container_width=True):
    try:
        pred, prob = predict_customer(customer)
        st.metric("Churn probability", f"{prob:.1%}")
        if pred == 1:
            st.error("High-risk prediction: the model predicts this customer may churn.")
        else:
            st.success("Lower-risk prediction: the model predicts this customer will stay.")
        st.progress(min(prob, 1.0))
    except FileNotFoundError:
        st.warning("Model file not found. Run `python src/train.py` first.")
