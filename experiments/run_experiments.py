import os
import time
from datetime import datetime

import pandas as pd

from models import gcforest, logistic_regression, random_forest, svm, xgb
from optimizers import ga, pso, sa
from utils.data_loader import load_data

# Create results directory if not exists
os.makedirs("results", exist_ok=True)


def run_experiments():
    X_train, X_val, y_train, y_val = load_data()

    models = [
        ("SVM", svm),
        ("RandomForest", random_forest),
        ("XGBoost", xgb),
        ("LogisticRegression", logistic_regression),
        ("GCForest", gcforest),
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
        results.append(
            {
                "Model": model_name,
                "Optimizer": "Baseline",
                "Accuracy": baseline_acc,
                "Params": model_module.default_params(),
                "Time": elapsed,
            }
        )

        for opt_name, optimizer in optimizers[1:]:
            print(f"Running {model_name} with {opt_name} optimizer...")
            start_time = time.time()
            best_model, best_params, best_score = optimizer(
                model_module, X_train, y_train
            )
            elapsed = time.time() - start_time
            val_acc = model_module.evaluate(best_model, X_val, y_val)
            results.append(
                {
                    "Model": model_name,
                    "Optimizer": opt_name,
                    "Accuracy": val_acc,
                    "Params": best_params,
                    "Time": elapsed,
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/experiment_results_{timestamp}.csv"

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Experiments completed. Results saved to {output_file}")


if __name__ == "__main__":
    run_experiments()
