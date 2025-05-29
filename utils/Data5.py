import kagglehub
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
    path = kagglehub.dataset_download("abhinavkrjha/fake-news-challenge")

    # Load all stance and body files
    stance_files = [
        "train_stances.csv",
        "competition_test_stances_unlabeled.csv",
        "test_stances_unlabeled.csv"
    ]

    body_files = [
        "train_bodies.csv",
        "competition_test_bodies.csv",
        "test_bodies.csv"
    ]

    # Load and concatenate stance files
    stances = pd.concat(
        [pd.read_csv(os.path.join(path, fname)) for fname in stance_files],
        ignore_index=True
    )

    # Load and concatenate body files
    bodies = pd.concat(
        [pd.read_csv(os.path.join(path, fname)) for fname in body_files],
        ignore_index=True
    )

    # Output structure confirmation
    print("Stances shape:", stances.shape)
    print("Bodies shape:", bodies.shape)
    print("Stances columns:", stances.columns)
    print("Bodies columns:", bodies.columns)

    # Only use labeled training data
    merged = pd.merge(stances, bodies, on="Body ID", how="left")
    merged.dropna(subset=["Headline", "articleBody", "Stance"], inplace=True)

    # Combine headline and body
    merged["headline_body"] = merged["Headline"].fillna("") + " " + merged["articleBody"].fillna("")
    X_raw = [clean_text(text) for text in merged["headline_body"]]

    # Map stances to numeric labels
    stance_map = {"unrelated": 0, "discuss": 1, "agree": 2, "disagree": 3}
    y = merged["Stance"].map(stance_map).values

    # Vectorize and feature selection
    X_tfidf, vectorizer = vectorize_texts(X_raw)
    selector = SelectKBest(score_func=chi2, k=10000)
    X_selected = selector.fit_transform(X_tfidf, y)

    print("raw:", len(X_raw))
    print("tfidf:", X_tfidf.shape)
    print("selected:", X_selected.shape)

    return X_selected, y
