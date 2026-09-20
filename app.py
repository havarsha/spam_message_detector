import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)
import base64

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(5, 20, 45, 0.72), rgba(5, 20, 45, 0.72)),
                url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main .block-container {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("assets/background.jpg")
# Title
st.title("📩 Spam Message Detector")

st.write(
    "Enter a message below and our machine learning model "
    "will predict whether it is Spam or Not Spam."
)

st.divider()

# Message input
message = st.text_area(
    "💬 Enter your message",
    placeholder="Example: Congratulations! You have won a free prize!",
    height=150
)

# Prediction button
if st.button("🔍 Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:
        # Convert message into TF-IDF features
        message_vector = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_vector)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(message_vector)[0]

        # Get confidence
        confidence = max(probabilities) * 100

        st.divider()

        # Display result
        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
            st.write("This message is likely to be spam.")

        else:
            st.success("✅ NOT SPAM")
            st.write("This message appears to be legitimate.")

        # Display confidence
        st.metric(
            label="🤖 Model Confidence",
            value=f"{confidence:.2f}%"
        )

        # Show probability breakdown
        st.write("### Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.write("✅ Not Spam")
            st.progress(float(probabilities[0]))
            st.write(f"{probabilities[0] * 100:.2f}%")

        with col2:
            st.write("🚨 Spam")
            st.progress(float(probabilities[1]))
            st.write(f"{probabilities[1] * 100:.2f}%")

# Example messages
st.divider()

st.subheader("🧪 Try an Example")

col1, col2 = st.columns(2)

with col1:
    st.write("**🚨 Spam Example**")
    st.code(
        "Congratulations! You have won a free prize. "
        "Click now to claim!"
    )

with col2:
    st.write("**✅ Normal Example**")
    st.code(
        "Hey, are you coming to college tomorrow?"
    )

# About section
st.divider()

with st.expander("ℹ️ About this project"):
    st.write(
        "This spam detection system uses machine learning to classify "
        "SMS messages as Spam or Not Spam."
    )

    st.write("**Technologies used:**")
    st.write(
        "- Python\n"
        "- Pandas\n"
        "- Scikit-learn\n"
        "- TF-IDF Vectorization\n"
        "- Multinomial Naive Bayes\n"
        "- Streamlit\n"
        "- Joblib"
    )