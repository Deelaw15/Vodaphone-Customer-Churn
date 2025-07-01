import streamlit as st
import numpy as np
import joblib


# Load models
structured_model = joblib.load(open("churn_model.pkl", "rb"))  # trained on structured data
text_model = joblib.load(open("vodafone_churn_model.pkl", "rb"))  # trained on review vectors
vectorizer = joblib.load(open("vodafone_vectorizer.pkl", "rb"))

# Title
st.title("📉 Vodafone Churn Risk Assistant")

# Input selection
input_type = st.radio("Select input type:", ["Structured Features", "Customer Review"])

if input_type == "Structured Features":
    st.subheader("Enter Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100)
    tenure = st.slider("Tenure (months)", min_value=0, max_value=60)
    monthly_charges = st.number_input("Monthly Charges")
    # Add all other features needed (up to 13 total)

    if st.button("Predict Churn"):
        # Build feature array
        input_features = np.array([age, tenure, monthly_charges, ...])  # complete with all 13
        input_features = input_features.reshape(1, -1)
        prediction = structured_model.predict(input_features)[0]
        st.success("Prediction: " + ("Churn" if prediction == 1 else "Not Churn"))

elif input_type == "Customer Review":
    st.subheader("Paste Customer Review")
    review = st.text_area("Customer Feedback")

    if st.button("Analyze Review for Churn Risk"):
        review_vector = vectorizer.transform([review])
        prediction = text_model.predict(review_vector)[0]
        st.success("Prediction from review: " + ("Churn" if prediction == 1 else "Not Churn"))
