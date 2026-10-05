import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from src.data_loader import load_data
from src.preprocessing import clean_text
from src.vectorizer import get_vectorizer
from src.model import get_model

def train():
    df = load_data()

    df["content"] = df["content"].apply(clean_text)

    X = df["content"]
    y = df["label"]

    pipeline = Pipeline([
        ("tfidf", get_vectorizer()),
        ("model", get_model())
    ])

    # 🔥 TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    pipeline.fit(X, y)
     # TEST PREDICTION
    y_pred = pipeline.predict(X_test)

    print("\n=== TEST SET EVALUATION ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(pipeline, "models/fake_news_pipeline.pkl")
    print("Model trained & saved")

if __name__ == "__main__":
    train()
