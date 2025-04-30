from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def create_model(params):
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=1337,
    )


def default_params():
    return {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }


def param_space():
    return [
        {"name": "n_estimators", "type": "continuous", "bounds": [50, 300]},
        {"name": "max_depth", "type": "continuous", "bounds": [5, 50]},
        {"name": "min_samples_split", "type": "continuous", "bounds": [2, 10]},
        {"name": "min_samples_leaf", "type": "continuous", "bounds": [1, 10]},
    ]


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)
