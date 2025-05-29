import os
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import kagglehub

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Define text cleaning function
def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in filtered_tokens]
    return " ".join(stemmed)

# Define sentiment classification
def compute_sentiment(text, sia):
    score = sia.polarity_scores(text)['compound']
    if score < 0:
        return 0.0
    elif score > 0:
        return 1.0
    else:
        return 0.5

# Identity tokenizer
def identity_tokenizer(text):
    return text.split()

# Define load_data function
def load_data():
    path = kagglehub.dataset_download("arunavakrchakraborty/covid19-twitter-dataset")

    files = [
        "Covid-19 Twitter Dataset (Apr-Jun 2020).csv",
        "Covid-19 Twitter Dataset (Apr-Jun 2021).csv",
        "Covid-19 Twitter Dataset (Aug-Sep 2020).csv"
    ]

    dfs = []
    for file in files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, encoding="utf-8")
        if "Original Tweet" not in df.columns:
            continue
        df = df[["Original Tweet"]].dropna()
        df["cleaned_tweet"] = df["Original Tweet"].apply(clean_text)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    sia = SentimentIntensityAnalyzer()
    combined_df["sentiment"] = combined_df["cleaned_tweet"].apply(lambda x: compute_sentiment(x, sia))

    X_raw = combined_df["cleaned_tweet"].tolist()
    y = combined_df["sentiment"].values

    vectorizer = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        use_idf=True,
        norm="l2",
        smooth_idf=True,
        ngram_range=(1, 2)
    )
    X_tfidf = vectorizer.fit_transform(X_raw)

    selector = SelectKBest(score_func=chi2, k=10000)
    X_selected = selector.fit_transform(X_tfidf, y)

    print("raw:", len(X_raw))
    print("tfidf:", X_tfidf.shape)
    print("selected:", X_selected.shape)

    return X_selected, y
