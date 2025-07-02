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
```

# 🖥️ How to Run Locally

Follow these steps to run the Fake News Detection project on your local machine.

## 📁 Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```
Or download the ZIP and extract it manually.

## 📥 Step 2: Download Required Files

Since large files are not included in this repo, download them from Google Drive:

👉 Download data/ folder and model files

Place them like this:
```
fake-news-detection/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submit.csv
├── xgb_model.pkl
├── xgb_vectorizer.pkl

```

## 🐍 Step 3: Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
## 📦 Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
Also, ensure you download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```
## 🧪 Step 5: Test Model Performance (Optional)

To evaluate all models:
```python
python test_runner.py
```
## 🚀 Step 6: Launch the Gradio App

This opens a local web interface in your browser:
```python
python gradio_app.py
```
You’ll see something like:
```nginx
Running on local URL: http://127.0.0.1:7860
```
Now, just enter any news title and content to test if it's real or fake!

