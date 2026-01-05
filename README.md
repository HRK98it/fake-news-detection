📰 Fake News Detection System (Diagnobot)

A multimodal fake news detection system that classifies news as Fake or Real using Machine Learning, supporting text, image, and video inputs through a unified pipeline.

🚀 Features

✅ Text-based Fake News Detection

🖼️ Image-based Detection using OCR

🎥 Video-based Detection using Speech-to-Text

📊 Confidence-based Prediction

🖥️ Interactive Streamlit Web Interface

🔁 Single ML Pipeline for all inputs

🧠 Project Motivation

Fake news spreads rapidly on social media platforms and can influence public opinion, elections, and societal harmony.
This project aims to automatically detect fake news by analyzing content from multiple media formats using Natural Language Processing (NLP) and Machine Learning.

🗂️ Project Structure
Diagnobot/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   └── fake_news_pipeline.pkl
│
├── notebook/
│   └── EDA.ipynb
│
├── src/
│   ├── train.py            # Model training
│   ├── predict.py          # Prediction logic
│   ├── preprocessing.py   # Text preprocessing
│   ├── ocr.py              # Image OCR module
│   └── video_transcript.py# Video speech-to-text
│
└── venv/                   # Virtual environment

📊 Dataset

Source: Kaggle Fake News Dataset

Files Used:

Fake.csv – Fake news articles

True.csv – Real news articles

Language: English

Type: News articles (political & social)

⚙️ Methodology

Data Preprocessing

Lowercasing

Punctuation removal

Stopword handling

Text normalization

Feature Extraction

TF-IDF Vectorization

Model Training

Logistic Regression

Implemented inside a Scikit-learn Pipeline

Prediction Strategy

Binary classification: Fake / Real

Confidence-based thresholding to reduce false positives

Multimodal Handling

Images: OCR → Text → Model

Videos: Speech-to-text → Model

🧪 Model Performance

Accuracy: ~95–99% (on test dataset)

Validation: Stratified train-test split

Note: High accuracy is dataset-specific; real-world behavior is handled using confidence thresholds.

🖥️ Web Application (Streamlit)

The system provides a user-friendly web interface where users can:

Paste text news

Upload image files

Upload video files

The app then displays:

Extracted text (for image/video)

Prediction result (Fake / Real)

Confidence score

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone <repo-link>
cd Diagnobot

2️⃣ Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
python -m pip install -r requirements.txt

4️⃣ Train the Model
python -m src.train

5️⃣ Run the Application
python -m streamlit run app.py

⚠️ Constraints & Limitations

OCR accuracy depends on image quality

Video prediction depends on audio clarity

Model is trained on news-style text

Informal or very short text may affect prediction confidence

System-level dependencies (OCR, audio processing) required for multimedia inputs

🔮 Future Enhancements

Integration with real-time social media feeds

Use of deep learning models (BERT, LSTM)

Multilingual fake news detection

Online deployment using Docker

Conclusion

This project demonstrates a practical and scalable approach to fake news detection using Machine Learning and NLP.
By supporting text, image, and video inputs, it showcases real-world applicability and strong engineering design.
