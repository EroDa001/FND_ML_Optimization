from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from experiments.config import RANDOM_SEED


def create_model(params):
    return LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        # solver="liblinear" if params["penalty"] == "l1" else "lbfgs",
        solver="saga",
        max_iter=1000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def default_params():
    return {"C": 1.0, "penalty": "l2"}


def param_space():
    return [
        {"name": "C", "type": "continuous", "bounds": [0.01, 10]},
        {"name": "penalty", "type": "categorical", "categories": ["l1", "l2"]},
    ]


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)
