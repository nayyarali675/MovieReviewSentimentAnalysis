import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import joblib
from src.preprocessing import clean_text

def train_model(model_type="logreg"):
    df = pd.read_csv("data/IMDB_Dataset.csv")
    df["cleaned"] = df["review"].apply(clean_text)
    
    X = df["cleaned"]
    y = df["sentiment"].map({"positive": 1, "negative": 0})

    tfidf = TfidfVectorizer(max_features=5000)
    X_vect = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    if model_type == "logreg":
        model = LogisticRegression()
    elif model_type == "nb":
        model = MultinomialNB()
    elif model_type == "svm":
        model = LinearSVC()
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{model_type.upper()} Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
