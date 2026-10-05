# Fake News Detection System

A machine-learning application that classifies news text as **Fake** or **Real** using a trained TF-IDF + Logistic Regression pipeline and a Streamlit web interface.

## Features

- Text-based fake news classification
- TF-IDF feature extraction
- Logistic Regression classifier
- Model score displayed in the Streamlit UI
- Input validation for very short text
- Ready for deployment on Render

## Project Structure

```text
fake-news-detection-main/
├── app.py
├── requirements.txt
├── runtime.txt
├── render.yaml
├── models/
│   └── fake_news_pipeline.pkl
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── kfold_validate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── vectorizer.py
├── dataset/
│   ├── Fake.csv
│   └── True.csv
└── notebook/
    └── EDA.ipynb
```

## Run Locally

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Deployment on Render

This repository includes `render.yaml`. You can connect the GitHub repository to Render as a Web Service.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Model

The production application loads the already-trained model from:

```text
models/fake_news_pipeline.pkl
```

The saved pipeline contains TF-IDF vectorization and Logistic Regression, so the application does not need to retrain the model when it starts.

## Important Note

The prediction represents the classification made by the trained model. It does not independently verify the factual accuracy of a news story and should not be treated as a definitive fact-checking service.
