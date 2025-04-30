import os

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    path = kagglehub.dataset_download(
        "free4ever1/instagram-fake-spammer-genuine-accounts"
    )
    if path is None:
        raise ValueError("Dataset not found. Please check the dataset name.")

    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")

    with open(train_path, "r") as file:
        train_dataframe = pd.read_csv(file, encoding="utf-8")

    with open(test_path, "r") as file:
        test_dataframe = pd.read_csv(file, encoding="utf-8")

    dataframe = pd.concat([train_dataframe, test_dataframe], ignore_index=True)

    dataframe.rename(
        columns={
            "profile pic": "has_profile_pic",
            "nums/length username": "username_nums_ratio",
            "fullname words": "fullname_words_count",
            "nums/length fullname": "fullname_nums_ratio",
            "name==username": "name_same_as_username",
            "description length": "bio_char_count",
            "external URL": "has_external_url",
            "private": "is_private",
            "#posts": "posts_count",
            "#followers": "followers_count",
            "#follows": "following_count",
            "fake": "is_fake",
        },
        inplace=True,
    )

    target = "is_fake"

    X = dataframe.drop(columns=[target])
    y = dataframe[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1337, stratify=y
    )

    return X_train, X_val, y_train, y_val
