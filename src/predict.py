from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_pipeline.pkl"

# Load the trained TF-IDF + Logistic Regression pipeline once when the app starts.
pipeline = joblib.load(MODEL_PATH)


MIN_WORDS = 8
FAKE_THRESHOLD = 0.75


def predict_news(text: str):
    """Predict whether news text is fake or real.

    Returns:
        (1, score) -> fake
        (0, score) -> real
        (None, 0.0) -> insufficient input
    """
    if not isinstance(text, str) or not text.strip():
        return None, 0.0

    text = text.strip()

    if len(text.split()) < MIN_WORDS:
        return None, 0.0

    fake_probability = float(pipeline.predict_proba([text])[0][1])

    if fake_probability >= FAKE_THRESHOLD:
        return 1, fake_probability

    return 0, 1.0 - fake_probability
