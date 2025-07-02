# =======================
# xgboost_model.py
# =======================

import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.preprocessing import clean_and_stem
import joblib



def train_xgboost(train_path="data/train.csv"):
    df = pd.read_csv(train_path).fillna("")
    df["content"] = df[['title', 'author', 'text']].astype(str).agg(' '.join, axis=1)
    df["content"] = df["content"].apply(clean_and_stem)

    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df["content"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "xgb_model.pkl")
    joblib.dump(vectorizer, "xgb_vectorizer.pkl")


    y_pred = model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
