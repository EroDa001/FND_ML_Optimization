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
    base_path = "/kaggle/input/fakeddit-dataset/multimodal_only_samples"
    train_path = f"{base_path}/multimodal_train.tsv"
    test_path = f"{base_path}/multimodal_test_public.tsv"
    val_path = f"{base_path}/multimodal_val.tsv"

    df_train = pd.read_csv(train_path, sep='\t', usecols=["clean_title", "2_way_label"])
    df_test = pd.read_csv(test_path, sep='\t', usecols=["clean_title", "2_way_label"])
    df_val = pd.read_csv(val_path, sep='\t', usecols=["clean_title", "2_way_label"])

    # Concatenate all splits
    df = pd.concat([df_train, df_test, df_val], ignore_index=True)

    # Drop missing
    df.dropna(subset=["clean_title", "2_way_label"], inplace=True)

    # Clean text
    X_raw = [clean_text(text) for text in df["clean_title"]]
    y = df["2_way_label"].astype(int).values

    # Vectorize
    X_tfidf, vectorizer = vectorize_texts(X_raw)

    # Feature selection
    selector = SelectKBest(score_func=chi2, k=10000)
    X_selected = selector.fit_transform(X_tfidf, y)

    print("raw:", len(X_raw))
    print("tfidf:", X_tfidf.shape)
    print("selected:", X_selected.shape)

    return X_selected, y
