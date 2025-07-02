import streamlit as st
import joblib   
import time

from gensim.models import LdaModel
from gensim.corpora import Dictionary

# === Load Models and Tools ===
model = joblib.load(open("vodafone_churn_model.pkl", "rb"))
vectorizer = joblib.load(open("vodafone_vectorizer.pkl", "rb"))
lda_model = LdaModel.load("lda_model.gensim")
dictionary = Dictionary.load("lda_dictionary.dict")

#lda_model = pickle.load(open("lda_model.gensim", "rb"))
#dictionary = joblib.load(open("lda_dictionary.dict", "rb"))

# === Topic Mapping & Retention Responses ===
topic_actions = {
    "Customer Service": "Recognize and reward outstanding support staff to improve satisfaction.",
    "Pricing & Charges": "Consider offering a discount or explaining charges to reduce confusion.",
    "Network & Coverage": "Investigate local network issues and offer service guarantees.",
    "Technical Support": "Follow up with technical help or offer premium support.",
    "Positive Feedback": "Send a thank-you message or loyalty reward to reinforce brand satisfaction.",
    "Contract & Terms": "Provide flexible contract options to improve retention.",
    "Default": "Thank the customer for their feedback and monitor for further signals."
}

# === Topic Label Mapping ===
topic_labels = {
    0: "Customer Service",
    1: "Pricing & Charges",
    2: "Network & Coverage",
    3: "Technical Support",
    4: "Positive Feedback",
    5: "Contract & Terms"
}

# === Streamlit Layout ===
st.markdown("## 📈 SmartRetain - Churn Prediction & Response System")
st.markdown("Welcome! Paste a customer review below, and SmartRetain will predict if the customer is likely to churn and suggest a personalized action.")

# === Input Area ===
review = st.text_area("✍️ Enter Customer Review")

if st.button("🔍 Analyze Review"):
    with st.spinner("Analyzing review..."):
        # Simulate wait time
        time.sleep(1.5)

        # Vectorize input
        review_vector = vectorizer.transform([review])

        # Predict churn
        churn_pred = model.predict(review_vector)[0]
        prediction = "Churn" if churn_pred == 1 else "No Churn"

        # Identify topic using LDA
        bow = dictionary.doc2bow(review.lower().split())
        topics = lda_model.get_document_topics(bow)
        if topics:
            top_topic_id = max(topics, key=lambda x: x[1])[0]
            topic = topic_labels.get(top_topic_id, "Other")
        else:
            topic = "Unknown"

        # Suggest Action
        action = topic_actions.get(topic, topic_actions["Default"])

    # === Display Output ===
    st.success(f"**Prediction:** {prediction}")
    st.info(f"**Topic Identified:** {topic}")
    st.warning(f"**⚠️ Suggested Action:** {action}")
