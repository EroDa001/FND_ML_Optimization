# Fake News Detection with Machine Learning

This repository contains a complete pipeline for fake news detection using classical machine learning algorithms, optimized with game-based metaheuristics.

## 📌 Overview

This project explores the use of traditional ML classifiers for detecting fake news in text data. It includes preprocessing, feature extraction, model training, evaluation, and hyperparameter tuning using advanced optimization techniques inspired by games.

## 🧠 Main Features

- **Text preprocessing** using NLTK and BeautifulSoup  
- **Lightweight vectorization** (alternative to TF-IDF)  
- **Multiple classifiers**: SVM, Logistic Regression, Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost), etc.  
- **Game-based optimization** of hyperparameters using:
  - Golden Ball Optimizer (GBO)
  - Dice Game Optimizer (DGO)
  - Hide and Seek Optimizer (HSO) [Mealpy Framework]
  - Search and Rescue Optimization (SARO) [Mealpy]
- **Extensive evaluation** with 9+ metrics:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC, Cohen’s Kappa, Balanced Accuracy
  - Log Loss, Confusion Matrix

## 🚀 Results

- Achieved **10% improvement over Google Gemini** on custom benchmark using optimized models.
- Identified best combinations of preprocessing, feature selection, and classifier via exhaustive testing.


## 🛠️ Requirements

- Python 3.9+
- scikit-learn
- pandas
- numpy
- nltk
- lightgbm, xgboost, catboost
- mealpy
- shap, matplotlib, seaborn

Install dependencies:
```bash
pip install -r requirements.txt


