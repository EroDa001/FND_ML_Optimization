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

def load_data():
    # Download dataset
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    fake_path = os.path.join(path, "Fake.csv")
    real_path = os.path.join(path, "True.csv")

    # Load CSVs
    df_fake = pd.read_csv(fake_path, encoding="utf-8")
    df_real = pd.read_csv(real_path, encoding="utf-8")
    print(df_fake.shape)
    print(df_real.shape)

    # Assign labels
    df_fake["label"] = 0
    df_real["label"] = 1

    # Concatenate datasets
    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Drop duplicates
    df.drop_duplicates(subset=["title", "text"], inplace=True)

    # Drop rows with missing content
    df.dropna(subset=["title", "text"], inplace=True)

    # Merge title and text for processing
    df["title_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Preprocessing
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"[0-9]", " ", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        tokens = text.lower().strip().split()
        filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return " ".join(filtered)

    X_raw = [clean_text(t) for t in df["title_text"]]
    y = df["label"].values

    # Define tokenizer for TfidfVectorizer
    def identity_tokenizer(text):
        return text.split()

    vectorizer = TfidfVectorizer(
        #max_features=10000,
        min_df=2,
        max_df=0.95,
        tokenizer=identity_tokenizer,
        use_idf=True,
        norm="l2",
        smooth_idf=True,
        ngram_range=(1, 2)
    )
    X_tfidf = vectorizer.fit_transform(X_raw)

    # Feature selection
    selector = SelectKBest(score_func=chi2, k=1000)
    X_selected = selector.fit_transform(X_tfidf, y)
    print(X_tfidf.shape)
    print(X_selected.shape)



    return X_selected, y
