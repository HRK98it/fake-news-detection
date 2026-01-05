from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_loader import load_data
from src.preprocessing import clean_text
from src.vectorizer import get_vectorizer
from src.model import get_model

def run_kfold():
    df = load_data()

    df["content"] = df["content"].apply(clean_text)

    X = df["content"]
    y = df["label"]

    pipeline = Pipeline([
        ("tfidf", get_vectorizer()),
        ("model", get_model())
    ])

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )

    print("Fold-wise Accuracy:", scores)
    print("Mean Accuracy:", scores.mean())
    print("Standard Deviation:", scores.std())

if __name__ == "__main__":
    run_kfold()
