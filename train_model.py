import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Loading the dataset

df = pd.read_csv('./data/abstracts_new.csv')

texts = df['abstract'].astype(str).tolist()
labels = df['category'].astype(str).tolist()

# Split into train/test (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
#building the pipeline

model_pipeline = Pipeline([
     ('vectorizer', CountVectorizer(stop_words='english', max_features=10000)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate (optional)
accuracy = model_pipeline.score(X_test, y_test)

print(f"Model trained with accuracy:{accuracy:.2f}")

joblib.dump(model_pipeline,"model/sci_model.pkl")
print(" Model pipeline saved to model/sci_model.pkl")
