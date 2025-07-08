# model_trainer.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # For saving/loading models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re  # Regular expressions for text cleaning

# --- 1. Define a small, in-memory dataset ---
# In a real-world scenario, you'd load this from a CSV, database, etc.
# For this assignment, we'll create a simple one for demonstrative purposes.
data = {
    "text": [
        "I love this product! It's amazing and works perfectly.",
        "This is a terrible experience. Very disappointing.",
        "The weather today is neutral, neither good nor bad.",
        "Fantastic service, truly exceptional!",
        "Absolutely horrible, never again.",
        "It's an interesting concept, but needs refinement.",
        "Feeling great and happy about the results.",
        "So sad to hear that news.",
        "The meeting concluded at 5 PM.",
        "Excellent quality and fast delivery.",
        "Worst purchase ever, completely broken.",
        "Just a normal day at the office.",
        "Highly recommend this movie, so engaging!",
        "Disliked everything about it, a total waste.",
        "The cat sat on the mat.",
        "This is simply the best!",
        "I am so angry with this situation.",
        "It was okay, nothing special.",
        "Thrilled with the outcome!",
        "What a nightmare, absolutely frustrating.",
        "The report contains various statistics.",
    ],
    "sentiment": [
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Neutral",
    ],
}

df = pd.DataFrame(data)

# --- 2. Text Preprocessing Function ---
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize text
    words = word_tokenize(text)
    # Remove stopwords and join back
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


print("Preprocessing text data...")
df["processed_text"] = df["text"].apply(preprocess_text)
print("Preprocessing complete.")

# --- 3. Split Data into Training and Testing Sets ---
X = df["processed_text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- 4. Feature Extraction (TF-IDF Vectorizer) ---
# TF-IDF converts text into a matrix of TF-IDF features.
print("Training TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=1000)  # Limit features to 1000 for simplicity
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("TF-IDF Vectorizer trained.")

# --- 5. Train the Sentiment Classification Model (Logistic Regression) ---
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
model.fit(X_train_vec, y_train)
print("Model trained successfully.")

# --- 6. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# --- 7. Save the Trained Vectorizer and Model ---
# We need to save both the vectorizer (to transform new input text)
# and the trained model (to make predictions).
model_filename = "sentiment_model.joblib"
vectorizer_filename = "tfidf_vectorizer.joblib"

joblib.dump(model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

print(f"\nModel saved to: {model_filename}")
print(f"Vectorizer saved to: {vectorizer_filename}")

print("\n--- Quick Test with Saved Model ---")
# Load the saved model and vectorizer for a quick test
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)

test_text_1 = "This is an absolutely fantastic system, I love it!"
test_text_2 = "I am so disappointed with the poor performance."
test_text_3 = "The car is red."

processed_test_text_1 = preprocess_text(test_text_1)
processed_test_text_2 = preprocess_text(test_text_2)
processed_test_text_3 = preprocess_text(test_text_3)

# Transform new text using the loaded vectorizer
test_vec_1 = loaded_vectorizer.transform([processed_test_text_1])
test_vec_2 = loaded_vectorizer.transform([processed_test_text_2])
test_vec_3 = loaded_vectorizer.transform([processed_test_text_3])

# Predict sentiment
prediction_1 = loaded_model.predict(test_vec_1)[0]
prediction_2 = loaded_model.predict(test_vec_2)[0]
prediction_3 = loaded_model.predict(test_vec_3)[0]

print(f"Text: '{test_text_1}' -> Sentiment: {prediction_1}")
print(f"Text: '{test_text_2}' -> Sentiment: {prediction_2}")
print(f"Text: '{test_text_3}' -> Sentiment: {prediction_3}")
