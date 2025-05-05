
# 🎬 IMDb Movie Review Sentiment Analysis

This project implements a sentiment analysis system to classify IMDb movie reviews as **positive** or **negative** using machine learning techniques. It was developed as part of an internship task focused on Natural Language Processing (NLP).

---

## 📁 Project Structure

```
movie_sentiment_analysis/
│
├── app/
│   └── app.py                        # Streamlit app for sentiment prediction
│
├── data/
│   └── IMDB_Dataset.csv              # IMDb movie review dataset (50,000 labeled samples)
│
├── models/
│   ├── sentiment_model.pkl          # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│
├── notebooks/
│   └── sentiment_analysis.ipynb     # Jupyter notebook with full pipeline and evaluation
│
├── reports/
│   └── sentiment_analysis_report.pdf  # Project summary and evaluation report
│
└── README.md                         # Project documentation
```

---

## 📌 Objectives

- Preprocess IMDb review data (lowercasing, punctuation removal, stopwords, tokenization)
- Train a Logistic Regression model using TF-IDF features
- Evaluate model performance using accuracy and F1-score
- (Optional) Provide a simple web interface using Streamlit

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- NLTK
- Pandas & NumPy
- Matplotlib & Seaborn
- Streamlit

---

## 🧹 Preprocessing Steps

1. Convert text to lowercase
2. Remove HTML tags and punctuation
3. Tokenize sentences using NLTK
4. Remove English stopwords

---

## 🤖 Model Details

- **Model**: Logistic Regression
- **Vectorizer**: TF-IDF (max 5000 features)
- **Accuracy**: ~89%
- **F1 Score**: ~89%
- **Output**: Positive (1) or Negative (0)

---

## 📊 Evaluation

- Confusion matrix plotted for model diagnostics
- Accuracy and F1-score metrics used for performance evaluation
- Predictions tested via Streamlit app

---

## ▶️ How to Run

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app/app.py
```

### 3. View the Notebook

Open `notebooks/sentiment_analysis.ipynb` for code, analysis, and visualizations.

---

## 📄 Final Report

Find the project report at `reports/sentiment_analysis_report.pdf` summarizing:
- Approach
- Challenges
- Evaluation metrics
- Future improvements

---

## 🚀 Future Enhancements

- Use deep learning (LSTM, BERT)
- Add neutral sentiment class
- Deploy app online using Streamlit Cloud or Render
- Add input validation and user feedback

---

## 👨‍💻 Author

Rawat — Internship Submission (May 2025)
