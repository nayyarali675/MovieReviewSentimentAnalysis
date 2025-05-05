import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.predict import predict_sentiment

st.title("🎬 Movie Review Sentiment Analyzer")
review = st.text_area("Enter your movie review:")

if st.button("Analyze"):
    if review.strip():
        result = predict_sentiment(review)
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter a review first.")
