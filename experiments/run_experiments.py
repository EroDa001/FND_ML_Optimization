import time
from datetime import datetime

import pandas as pd

from experiments.config import CV_FOLDS
from models import (adaboost, knn, logistic_regression, mlp, naive_bayes,
                    random_forest, svm, xgb)
from optimizers import ga, pso, sa
from utils.data_loader import load_data
from utils.metrics import compute_metrics


def run_experiments():
    X_train, y_train, X_val, y_val = load_data()

    models = [
        ("SVM", svm),
        ("RandomForest", random_forest),
        ("XGBoost", xgb),
        ("LogisticRegression", logistic_regression),
        ("AdaBoost", adaboost),
        ("KNN", knn),
        ("MLP", mlp),
        ("NaiveBayes", naive_bayes),
    ]
    optimizers = [
        ("Baseline", None),
        ("GA", ga.optimize),
        ("PSO", pso.optimize),
        ("SA", sa.optimize),
    ]

    results = []

    for model_name, model_module in models:
        print(f"Running {model_name} with default parameters, no optimizer...")
        start_time = time.time()
        baseline_model = model_module.create_model(model_module.default_params())
        baseline_model.fit(X_train, y_train)
        baseline_acc = model_module.evaluate(baseline_model, X_val, y_val)
        elapsed = time.time() - start_time

        y_pred = baseline_model.predict(X_val)
        y_prob = (
            baseline_model.predict_proba(X_val)
            if hasattr(baseline_model, "predict_proba")
            else None
        )

        base_metrics = compute_metrics(y_val, y_pred, y_prob)

        results.append(
            {
                "Model": model_name,
                "Optimizer": "Baseline",
                # "Accuracy": baseline_acc,
                **base_metrics,
                "Params": model_module.default_params(),
                "Time": elapsed,
            }
        )

        for opt_name, optimizer in optimizers[1:]:
            print(f"Running {model_name} with {opt_name} optimizer...")
            start_time = time.time()
            best_model, best_params, best_score = optimizer(
                model_module, X_train, y_train, cv=CV_FOLDS
            )
            elapsed = time.time() - start_time
            val_acc = model_module.evaluate(best_model, X_val, y_val)

            y_pred = best_model.predict(X_val)
            y_prob = (
                best_model.predict_proba(X_val)
                if hasattr(best_model, "predict_proba")
                else None
            )
            m = compute_metrics(y_val, y_pred, y_prob)

            results.append(
                {
                    "Model": model_name,
                    "Optimizer": opt_name,
                    **m,
                    "Params": best_params,
                    "Time": elapsed,
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/logs/experiment_results_{timestamp}.csv"

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Experiments completed. Results saved to {output_file}")


if __name__ == "__main__":
    run_experiments()
