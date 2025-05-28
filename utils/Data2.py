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
        #max_features=512,
        tokenizer=identity_tokenizer,
        use_idf=True,
        norm="l2",
        smooth_idf=True,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def load_data():
    data_path = "/home/moodyblues/Desktop/Projects Z/master/data/data2.csv"
    df = pd.read_csv(data_path, encoding="utf-8")

    # Drop the subcategory column
    if "subcategory" in df.columns:
        df.drop(columns=["subcategory"], inplace=True)

    # Drop rows with missing title or text
    df.dropna(subset=["title", "text"], inplace=True)
    df.drop_duplicates(subset=["title", "text"], inplace=True)

    # Combine title and text
    df["title_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Clean the text
    X_raw = [clean_text(t) for t in df["title_text"]]
    y = df["label"].values

    # Vectorize
    X_tfidf, vectorizer = vectorize_texts(X_raw)

    # Feature selection
    selector = SelectKBest(score_func=chi2, k=10000)
    X_selected = selector.fit_transform(X_tfidf, y)

    print("raw:", len(X_raw))
    print("tfidf:", X_tfidf.shape)
    print("selected:", X_selected.shape)

    return X_selected, y
