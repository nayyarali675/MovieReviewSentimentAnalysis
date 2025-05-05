import joblib
from src.preprocessing import clean_text

def predict_sentiment(text):
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)

    return "Positive" if prediction[0] == 1 else "Negative"
