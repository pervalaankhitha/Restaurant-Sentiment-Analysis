
import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

setup_nltk()

# Load trained model and vectorizer
@st.cache_resource
def load_model():
    artifacts = joblib.load("sentiment_model.pkl")
    return artifacts["model"], artifacts["vectorizer"]

model, vectorizer = load_model()

# Text preprocessing
stop_words = set(stopwords.words("english"))

# Keep important negation words
stop_words = stop_words - {"not", "no", "nor", "never"}

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Website settings
st.set_page_config(
    page_title="Restaurant Sentiment Analysis",
    page_icon="🍽️",
    layout="centered"
)

# Website heading
st.title("🍽️ Restaurant Review Sentiment Analysis")

st.write(
    "Enter a restaurant review below to predict "
    "whether the sentiment is Positive or Negative."
)

# User input
review = st.text_area(
    "Enter your restaurant review:",
    placeholder="Example: The food was fantastic!"
)

# Prediction button
if st.button("Analyze Sentiment"):

    if not review.strip():

        st.warning("Please enter a review.")

    else:

        # Clean review
        cleaned_review = clean_text(review)

        # Convert text into numerical features
        features = vectorizer.transform([cleaned_review])

        # Predict sentiment
        prediction = model.predict(features)[0]

        # Calculate confidence
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100

        # Display result
        if prediction == 1:

            st.success("😊 Positive Review")

        else:

            st.error("😞 Negative Review")

        st.write(f"Confidence: {confidence:.2f}%")

        st.write("Original Review:", review)

        st.write("Cleaned Review:", cleaned_review)

st.markdown("---")

st.write("Developed using Python, NLP and Naive Bayes.")
