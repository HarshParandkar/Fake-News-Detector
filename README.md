# 📰 Fake News Detection Using Machine Learning

This project uses multiple machine learning algorithms to detect fake news based on news **title**, **author**, and **text content**. It provides both a training pipeline for multiple models and a Gradio-based interactive UI for real-time testing.

---

## 📌 Features

- Text preprocessing (stopwords removal, stemming, lemmatization)
- TF-IDF vectorization
- Models used:
  - Logistic Regression
  - Random Forest
  - Multinomial Naive Bayes
  - Decision Tree
  - XGBoost ✅ *(used in final UI)*
  - Support Vector Machine (SVM)
- Gradio interface for live fake news detection
- Modular code structure with easy extensibility
- Exported model and vectorizer for deployment

---

## 📁 Dataset & Model Downloads

Since large files can't be uploaded to GitHub, download the required data and model files from the link below:

👉 [Download `data/` folder and model files from Google Drive](https://drive.google.com/drive/folders/1S60sfpEPN-ORo9NzRDoRImKJQ3ao6_Qm?usp=sharing)

After downloading, place them like this:

```bash
fake-news-detection/
├── data/                            # Dataset files (not on GitHub)
│   ├── train.csv
│   ├── test.csv
│   └── submit.csv
│
├── models/                          # ML model training scripts
│   ├── logistic_regression_model.py
│   ├── random_forest_model.py
│   ├── naive_bayes_model.py
│   ├── decision_tree_model.py
│   ├── xgboost_model.py
│   └── svm_model.py
│
├── utils/                           # Text preprocessing utilities
│   └── preprocessing.py
│
├── xgb_model.pkl                    # Trained XGBoost model (kept in Drive, not GitHub)
├── xgb_vectorizer.pkl               # TF-IDF vectorizer for the model
│
├── gradio_app.py                    # Gradio UI app (main interface)
├── test_runner.py                   # Script to compare all model performances
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitattributes
