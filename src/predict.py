import joblib

pipeline = joblib.load("models/fake_news_pipeline.pkl")

def predict_news(text):

    # short / casual text => REAL
    if len(text.split()) < 8:
        return 0, 1.0

    fake_prob = pipeline.predict_proba([text])[0][1]

    # 🔥 STRICT FAKE THRESHOLD
    if fake_prob >= 0.75:
        return 1, fake_prob        # FAKE
    else:
        return 0, 1 - fake_prob    # REAL
