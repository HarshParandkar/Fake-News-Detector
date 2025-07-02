# =======================
# logistic_regression_model.py
# =======================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils.preprocessing import clean_and_stem

def train_logistic_regression(train_path="data/train.csv"):
    df = pd.read_csv(train_path).fillna("")
    df["content"] = df[['title', 'author', 'text']].astype(str).agg(' '.join, axis=1)
    df["content"] = df["content"].apply(clean_and_stem)

    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df["content"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    param_grid = {'C': [0.01, 0.1, 1], 'penalty': ['l1', 'l2']}
    grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=3)
    grid.fit(X_train, y_train)

    best_model = LogisticRegression(solver='liblinear', **grid.best_params_)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)

    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    return best_model, vectorizer

