msg = ["Free money!!! click now"]# Import libraries
import pandas as pd
import numpy as np

# Text processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Model
from sklearn.naive_bayes import MultinomialNB

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Feature extraction (Text → Numbers)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# Test custom message
msg = ["Free money!!! click now"]
msg_vec = vectorizer.transform(msg)
print("\nPrediction:", model.predict(msg_vec))