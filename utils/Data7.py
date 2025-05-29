import os
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import kagglehub

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
    # Download dataset using kagglehub
    path = kagglehub.dataset_download("mohammadaflahkhan/fake-news-dataset-combined-different-sources")
    file_path = os.path.join(path, "PreProcessedData.csv")
    
    # Load CSV
    df = pd.read_csv(file_path)

    # Drop the index column if exists
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Drop missing and duplicate data
    df.dropna(subset=["title", "text", "Ground Label"], inplace=True)
    df.drop_duplicates(subset=["title", "text"], inplace=True)

    # Combine title and text
    df["title_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Encode labels: 1 for real, 0 for fake
    df["label"] = df["Ground Label"].apply(lambda x: 1 if str(x).strip().lower() == "real" else 0)

    # Clean combined text
    X_raw = [clean_text(t) for t in df["title_text"]]
    y = df["label"].values

    # TF-IDF vectorization
    X_tfidf, vectorizer = vectorize_texts(X_raw)

    # Feature selection
    selector = SelectKBest(score_func=chi2, k=min(10000, X_tfidf.shape[1]))
    X_selected = selector.fit_transform(X_tfidf, y)

    print("Raw samples:", len(X_raw))
    print("TF-IDF shape:", X_tfidf.shape)
    print("Selected shape:", X_selected.shape)

    return X_selected, y

