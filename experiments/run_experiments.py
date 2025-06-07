import time
from datetime import datetime

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from models import (
    logistic_regression, naive_bayes, random_forest,
    svm, xgb, catboost , lightgbm 
)
from optimizers import gbo, dgo, saro, hso
from utils.data_loader import load_data
from utils.metrics import compute_metrics


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def run_experiments():
    dataset = "Data4"
    (
        X_final_train,
        y_final_train,
        X_opt_train,
        y_opt_train,
        X_opt_val,
        y_opt_val,
        X_final_test,
        y_final_test,
    ) = load_data(dataset)

    models = [
        ("lightgbm", lightgbm),
        ("CatBoost", catboost),
        ("NaiveBayes", naive_bayes),
        ("RandomForest", random_forest),
        ("XGBoost", xgb),
        ("LogisticRegression", logistic_regression),
        ("SVM", svm),#todo
    ]

    optimizers = [
        ("Baseline", None),
        ("GBO", gbo.optimize),
        ("HSO", hso.optimize),
        ("SARO", saro.optimize),
        ("DGO", dgo.optimize),
    ]

    results = []

    for model_name, model_module in models:
        print(f"Running {model_name} with default parameters, no optimizer...")

        
        if model_name == "DeepForest":
            X_opt_train = to_dense(X_opt_train)
            X_opt_val = to_dense(X_opt_val)
            X_final_test = to_dense(X_final_test)

        time_start = time.time()
        baseline_model = model_module.create_model(model_module.default_params())
        baseline_model.fit(X_opt_train, y_opt_train)
        time_taken = time.time() - time_start

        print(
            f"Running {model_name} with default parameters, no optimizer... done in {time_taken:.2f} seconds"
        )

        for split_name, X, y in [
            ("Train", X_opt_train, y_opt_train),
            ("Test", X_final_test, y_final_test),
        ]:
            X = to_dense(X) if model_name == "DeepForest" else X
            y_pred = baseline_model.predict(X)
            y_prob = (
                baseline_model.predict_proba(X)
                if hasattr(baseline_model, "predict_proba")
                else None
            )
            mets = compute_metrics(y, y_pred, y_prob)
            print(f"{model_name} | Baseline | {split_name} metrics: {mets}")
            results.append(
                {
                    "Model": model_name,
                    "Dataset": dataset,
                    "Optimizer": "Baseline",
                    "Split": split_name,
                    **mets,
                    "Time": time_taken,
                    "Params": model_module.default_params(),
                }
            )

        for opt_name, optimizer in optimizers[1:]:
            print(f"Running {model_name} with {opt_name} optimizer...")
            time_start = time.time()

            best_model, best_params, best_score = optimizer(
                model_module,
                to_dense(X_opt_train) if model_name == "DeepForest" else X_opt_train,
                y_opt_train,
                to_dense(X_opt_val) if model_name == "DeepForest" else X_opt_val,
                y_opt_val,
            )

            time_end = time.time()
            time_taken = time_end - time_start
            print(
                f"Running {model_name} with {opt_name} optimizer... done in {time_taken:.2f} seconds"
            )

            for split_name, X, y in [
                ("Train", X_opt_train, y_opt_train),
                ("Test", X_final_test, y_final_test),
            ]:
                X = to_dense(X) if model_name == "DeepForest" else X
                y_pred = best_model.predict(X)
                y_prob = (
                    best_model.predict_proba(X)
                    if hasattr(best_model, "predict_proba")
                    else None
                )
                mets = compute_metrics(y, y_pred, y_prob)
                print(f"{model_name} | {opt_name} | {split_name} metrics: {mets}")
                results.append(
                    {
                        "Model": model_name,
                        "Dataset": dataset,
                        "Optimizer": opt_name,
                        "Split": split_name,
                        **mets,
                        "Time": time_taken,
                        "Params": best_params,
                    }
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/logs/experiment_results_{dataset}_{timestamp}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Experiments completed. Results saved to {output_file}")


if __name__ == "__main__":
    run_experiments()
