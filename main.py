import streamlit as st
import pandas as pd
import pickle

# Load the model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    loaded_model = model_data["model"]
    feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their details.")

# Input form
st.sidebar.header("Customer Details")
input_data = {}

# Add input fields for each feature
input_data['gender'] = st.sidebar.selectbox("Gender", ["Female", "Male"])
input_data['SeniorCitizen'] = st.sidebar.selectbox("Senior Citizen", [0, 1])
input_data['Partner'] = st.sidebar.selectbox("Partner", ["Yes", "No"])
input_data['Dependents'] = st.sidebar.selectbox("Dependents", ["Yes", "No"])
input_data['tenure'] = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
input_data['PhoneService'] = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
input_data['MultipleLines'] = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
input_data['InternetService'] = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
input_data['OnlineSecurity'] = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
input_data['OnlineBackup'] = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
input_data['DeviceProtection'] = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
input_data['TechSupport'] = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
input_data['StreamingTV'] = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
input_data['StreamingMovies'] = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
input_data['Contract'] = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
input_data['PaperlessBilling'] = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
input_data['PaymentMethod'] = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
input_data['MonthlyCharges'] = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=29.85)
input_data['TotalCharges'] = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=29.85)

# Convert input data to DataFrame
input_data_df = pd.DataFrame([input_data])

# Encode categorical features using the saved encoders
for column, encoder in encoders.items():
    if column in input_data_df.columns:
        input_data_df[column] = encoder.transform(input_data_df[column])

# Make a prediction
if st.sidebar.button("Predict Churn"):
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.write(f"**Probability of Churn:** {pred_prob[0][1]:.2f}")
    st.write(f"**Probability of No Churn:** {pred_prob[0][0]:.2f}")

# Display input data
st.subheader("Input Data")
st.write(input_data_df)