import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# LOAD FILES
# -----------------------------
model = joblib.load("customer_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("📊 Customer Churn Prediction")

st.write("Fill customer details:")

# -----------------------------
# USER INPUTS
# -----------------------------
tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# -----------------------------
# ENCODING MAPS
# -----------------------------
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

# -----------------------------
# CREATE FULL INPUT DATAFRAME
# -----------------------------
input_data = pd.DataFrame(columns=feature_columns)

# Fill all values with 0 first
input_data.loc[0] = 0

# Now update real values
input_data.at[0, "tenure"] = tenure
input_data.at[0, "MonthlyCharges"] = monthly
input_data.at[0, "TotalCharges"] = total
input_data.at[0, "Contract"] = contract_map[contract]
input_data.at[0, "InternetService"] = internet_map[internet]
input_data.at[0, "PaymentMethod"] = payment_map[payment]

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):

    try:
        # Apply scaling
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Result")

        if prediction[0] == 1:
            st.error(f"⚠️ Customer is likely to churn\n\nProbability: {probability:.2f}")
        else:
            st.success(f"✅ Customer will stay\n\nProbability: {probability:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")