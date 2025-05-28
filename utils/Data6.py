import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Define cleaning and sentiment functions
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

def compute_sentiment(text, sia):
    score = sia.polarity_scores(text)['compound']
    if score < 0:
        return 0.0
    elif score > 0:
        return 1.0
    else:
        return 0.5

# Load and preprocess dataset
def preprocess_covid_tweets():
    # Replace 'your_directory_path' with the path where your CSV files are located
    base_path = 'your_directory_path'
    files = [
        "Covid-19 Twitter Dataset (Apr-Jun 2020).csv",
        "Covid-19 Twitter Dataset (Apr-Jun 2021).csv",
        "Covid-19 Twitter Dataset (Aug-Sep 2020).csv"
    ]
    
    dfs = []
    for file in files:
        file_path = os.path.join(base_path, file)
        df = pd.read_csv(file_path, encoding="utf-8")
        df = df[["Original Tweet"]].dropna()
        df["cleaned_tweet"] = df["Original Tweet"].apply(clean_text)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    sia = SentimentIntensityAnalyzer()
    combined_df["sentiment"] = combined_df["cleaned_tweet"].apply(lambda x: compute_sentiment(x, sia))
    
    return combined_df

# Run the preprocessing
covid_tweets_df = preprocess_covid_tweets()
print(covid_tweets_df.head())
