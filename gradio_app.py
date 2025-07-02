# gradio_app.py

import gradio as gr
import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from utils.preprocessing import clean_and_stem

# Load model and vectorizer (make sure you've saved them!)
model = joblib.load("xgb_model.pkl")
vectorizer = joblib.load("xgb_vectorizer.pkl")

def predict_fake_news(text):
    processed = clean_and_stem(text)
    vect_text = vectorizer.transform([processed])
    pred = model.predict(vect_text)[0]
    return "🚫 FAKE NEWS" if pred == 1 else "✅ REAL NEWS"

interface = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=5, placeholder="Enter news article..."),
    outputs=gr.Label(),
    title="📰 Fake News Detector (XGBoost)",
    description="Enter the title or content of a news article to check whether it's fake or real."
)

if __name__ == "__main__":
    interface.launch()
