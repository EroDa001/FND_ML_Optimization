import os
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.lower().strip().split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(filtered)

def vectorize_texts(texts):
    def identity_tokenizer(text):
        return text.split()

    vectorizer = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        use_idf=True,
        norm="l2",
        smooth_idf=True,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def load_data():
    base_path = "/home/moodyblues/Desktop/Projects Z/master/data/data4"
    files = [
        "Constraint_English_Train - Sheet1.csv",
        "Constraint_English_Val - Sheet1.csv",
        "english_test_with_labels - Sheet1.csv"
    ]

    all_X_raw = []
    all_y = []

    for file in files:
        path = os.path.join(base_path, file)
        df = pd.read_csv(path)
        df.dropna(subset=["tweet", "label"], inplace=True)
        df.drop_duplicates(subset=["tweet"], inplace=True)
        X_raw = [clean_text(t) for t in df["tweet"]]
        y_raw = df["label"].map({"fake": 0, "real": 1}).values
        all_X_raw.extend(X_raw)
        all_y.extend(y_raw)

    # Vectorization
    X_tfidf, vectorizer = vectorize_texts(all_X_raw)

    # Feature selection
    selector = SelectKBest(score_func=chi2, k=10000)
    X_selected = selector.fit_transform(X_tfidf, all_y)

    print("raw:", len(all_X_raw))
    print("tfidf:", X_tfidf.shape)
    print("selected:", X_selected.shape)

    return X_selected, np.array(all_y)
