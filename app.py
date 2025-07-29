# app.py
from flask import Flask, render_template, request, jsonify

# ... (rest of your code) ...

# joblib is no longer needed as we are not loading a custom-trained model
# re (regex) is still useful for basic cleaning, but VADER handles some itself
# nltk.corpus.stopwords and nltk.tokenize.word_tokenize are also less critical for VADER's core logic,
# but keeping them in preprocess_text ensures consistent cleaning if you switch back or extend.
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Import VADER's SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os  # Still useful for checking file paths if needed for other features

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Initialize VADER Sentiment Analyzer ---
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool
# that is specifically attuned to sentiments expressed in social media.
analyzer = SentimentIntensityAnalyzer()

# --- Text Preprocessing Function ---
# VADER is robust to some noise, but basic cleaning is still good practice.
# Note: stopwords and tokenization are less critical for VADER itself,
# but keeping this function allows for consistent preprocessing.
stop_words = set(stopwords.words("english"))  # Ensure NLTK stopwords are downloaded


def preprocess_text(text):
    """
    Cleans text for sentiment analysis. VADER is robust, but basic cleaning helps.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove URLs (common in social media)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove user mentions (@username)
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags (keep text, remove #)
    text = re.sub(r"#", "", text)
    # Remove punctuation (keep apostrophes for contractions)
    text = re.sub(r"[^\w\s\']", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Optional: Remove stopwords and tokenize if you want to be very strict,
    # but VADER often works well with raw text as it handles common words.
    # For this VADER implementation, we'll keep it simpler and let VADER handle word weighting.

    return text


# --- Flask Routes ---


@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")


@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    """
    API endpoint to receive text, analyze its sentiment using VADER, and return the result.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text_to_analyze = data.get("text", "")

    if not text_to_analyze:
        return jsonify({"error": "No text provided for analysis"}), 400

    # Preprocess the input text
    cleaned_text = preprocess_text(text_to_analyze)

    # Use VADER to get sentiment scores
    # polarity_scores returns a dictionary with 'neg', 'neu', 'pos', and 'compound' scores.
    # 'compound' is a normalized, weighted composite score.
    sentiment_scores = analyzer.polarity_scores(cleaned_text)

    # Determine the overall sentiment based on the compound score
    # Common thresholds for VADER:
    # compound >= 0.05: Positive
    # compound <= -0.05: Negative
    # (anything in between is Neutral)
    compound_score = sentiment_scores["compound"]
    sentiment_prediction = "Neutral"
    if compound_score >= 0.05:
        sentiment_prediction = "Positive"
    elif compound_score <= -0.05:
        sentiment_prediction = "Negative"

    return jsonify(
        {
            "original_text": text_to_analyze,
            "sentiment": sentiment_prediction,
            "scores": sentiment_scores,  # Optional: show detailed scores for debugging/understanding
        }
    )


# --- Run the Flask App ---
if __name__ == "__main__":
    # No custom model to load, VADER is initialized directly.
    # You might want to add a print statement to confirm VADER is ready.
    print("VADER Sentiment Analyzer is ready.")
    # Run the Flask development server
    app.run(debug=True)  # debug=True allows auto-reloading and shows errors in browser
