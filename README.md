#  SciCheck: Research Abstract Classifier

SciCheck is a lightweight web app that classifies scientific abstracts into disciplines like Physics, Biology, or Computer Science using machine learning models (Naive Bayes and Logistic Regression).

## 🔍 Features
- Predict abstract category interactively
- Upload CSV to classify in bulk
- View classification report and confusion matrix
- Compare models

## 🚀 Try It
[Click to open Streamlit app](https://your-app-url)

## 🛠️ Tech Stack
- Python, scikit-learn
- Streamlit
- TfidfVectorizer, Logistic Regression, Naive Bayes

## 📦 Run Locally

```bash
pip install -r requirements.txt
streamlit run utils/main.py
