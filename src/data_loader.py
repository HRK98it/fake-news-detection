import pandas as pd

def load_data():
    fake = pd.read_csv("dataset/Fake.csv")
    real = pd.read_csv("dataset/True.csv")

    fake["label"] = 1
    real["label"] = 0

    df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
    df["content"] = df["title"] + " " + df["text"]

    return df
