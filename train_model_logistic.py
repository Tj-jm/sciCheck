import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# === Load your dataset ===
df = pd.read_csv("data/abstracts_new.csv")  # Make sure this file exists
assert "abstract" in df.columns and "category" in df.columns, "Columns 'abstract' and 'category' must be present"

# === Prepare data ===
X = df["abstract"]
y = df["category"]

# === Split for evaluation (optional but good practice) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Create pipeline ===
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.9)),
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto"))
])

# === Train ===
model_pipeline.fit(X_train, y_train)

# === Evaluate ===
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# === Save model ===
joblib.dump(model_pipeline, "./model/logistic_model.pkl")
print("✅ Model saved to ./model/logistic_model.pkl")
