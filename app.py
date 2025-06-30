import streamlit as st
import pickle

# Load your churn model and vectorizer
model = pickle.load(open("churn_model.pkl", "rb"))
vectorizer = pickle.load(open("vodafone_vectorizer.pkl", "rb"))

# Set page layout
st.set_page_config(page_title="SmartRetain Assistant", layout="centered")

# Title
st.title("📱 Vodafone Churn Risk Assistant")

# Chat input from user
user_input = st.chat_input("How can we help you today?")

# Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Predict and respond
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Vectorize and predict
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]
    
    # Generate response
    if prediction == 1:
        response = "⚠️ This customer is **likely to churn**. Recommend offering a retention promo."
    else:
        response = "✅ This customer seems satisfied. No immediate action needed."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()
