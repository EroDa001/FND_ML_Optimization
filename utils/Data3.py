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

columns = [
    'id', 'label', 'text', 'subject', 'speaker', 'job title',
    'state info', 'party', 'barely true', 'false', 'half true',
    'mostly true', 'pants on fire', 'context'
]

label_map = {
    'pants-fire': -3,
    'false': -2,
    'barely-true': -1,
    'half-true': 1,
    'mostly-true': 2,
    'true': 3
}

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
        # max_features=512,
        tokenizer=identity_tokenizer,
        use_idf=True,
        norm="l2",
        smooth_idf=True,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def load_data():
    path = kagglehub.dataset_download("mrigendraagrawal/liar-dataset")
    train_path = os.path.join(path, "liar_dataset", "train.tsv")
    valid_path = os.path.join(path, "liar_dataset", "valid.tsv")
    test_path  = os.path.join(path, "liar_dataset", "test.tsv")

    train = pd.read_csv(train_path, sep='\t', header=None, names=columns)
    valid = pd.read_csv(valid_path, sep='\t', header=None, names=columns)
    test  = pd.read_csv(test_path,  sep='\t', header=None, names=columns)

    df = pd.concat([train, valid, test], ignore_index=True)
    df.dropna(subset=["text", "label"], inplace=True)
    df["label"] = df["label"].map(label_map)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    X_raw = [clean_text(t) for t in df["text"]]
    y = df["label"].values

    X_tfidf, vectorizer = vectorize_texts(X_raw)

    selector = SelectKBest(score_func=chi2, k=10000)
    X_selected = selector.fit_transform(X_tfidf, y)

    print("raw:", len(X_raw))
    print("tfidf:", X_tfidf.shape)
    print("selected:", X_selected.shape)

    return X_selected, y
