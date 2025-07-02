import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_and_stem(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        return ' '.join([stemmer.stem(word) for word in text if word not in stop_words])
    return ""
