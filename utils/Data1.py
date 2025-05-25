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
#nltk.download("stopwords")
#nltk.download("wordnet")
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
        max_features=512,
        tokenizer=identity_tokenizer,
        use_idf=True,
        norm="l2",
        smooth_idf=True,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def load_data():
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    fake_path = os.path.join(path, "Fake.csv")
    real_path = os.path.join(path, "True.csv")

    df_fake = pd.read_csv(fake_path, encoding="utf-8")
    df_real = pd.read_csv(real_path, encoding="utf-8")

    df_fake["label"] = 0
    df_real["label"] = 1
    df = pd.concat([df_fake, df_real], ignore_index=True)
    df.drop_duplicates(subset=["title", "text"], inplace=True)
    df.dropna(subset=["title", "text"], inplace=True)
    df["title_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    X_raw = [clean_text(t) for t in df["title_text"]]
    y = df["label"].values

    X_tfidf, vectorizer = vectorize_texts(X_raw)

    selector = SelectKBest(score_func=chi2, k=512)
    X_selected = selector.fit_transform(X_tfidf, y)

    print("raw:", len(X_raw))
    print("tfidf:", X_tfidf.shape)
    print("selected:", X_selected.shape)

    return X_selected, y
