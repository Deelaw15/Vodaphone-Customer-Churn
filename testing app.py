import streamlit as st

# Set page config
st.set_page_config(page_title="SmartRetain", page_icon="🤖", layout="centered")

st.title("📱 SmartRetain - Customer Churn Assistant")

# Chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Simulated bot response (for now)
    response = "Thanks for your message! We're checking your churn risk..."

    # Save bot message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display bot message
    with st.chat_message("assistant"):
        st.markdown(response)
