import kagglehub
import os
import pandas as pd
import numpy as np

from sklearn.utils import shuffle

def load_data():
    # Download dataset
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    fake_path = os.path.join(path, "Fake.csv")
    real_path = os.path.join(path, "True.csv")

    # Load CSVs
    fake_df = pd.read_csv(fake_path, encoding="utf-8")
    real_df = pd.read_csv(real_path, encoding="utf-8")

    # Assign labels: 0 for fake, 1 for real
    fake_df["label"] = 0
    real_df["label"] = 1

    # Concatenate
    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Drop duplicates
    df.drop_duplicates(subset=["title", "text"], inplace=True)

    # Combine title and text into a single feature
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower()

    # Drop unused columns
    df.drop(columns=["title", "text", "subject", "date"], errors="ignore", inplace=True)

    # Shuffle
    df = shuffle(df, random_state=42).reset_index(drop=True)

    # Extract features and labels
    X = df["content"].values
    y = df["label"].values

    return X, y