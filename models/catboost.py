from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

from experiments.config import RANDOM_SEED


def create_model(params):
    return CatBoostClassifier(
        iterations=int(params["iterations"]),
        depth=int(params["depth"]),
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        rsm=params["colsample_bytree"],  # equivalent to colsample_bytree in CatBoost
        loss_function='MultiClass',
        bootstrap_type='Bernoulli',  # required to support subsample
        random_seed=RANDOM_SEED,
        verbose=0,
        thread_count=-1,
    )



def default_params():
    return {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.15,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
    }


def param_space():
    return [
        {"name": "iterations", "type": "categorical", "categories": list(range(25, 50, 75))},
        {"name": "depth", "type": "categorical", "categories": list(range(3, 5))},
        {"name": "learning_rate", "type": "continuous", "bounds": [0.1, 0.3]},
        {"name": "subsample", "type": "continuous", "bounds": [0.5, 0.8]},
        {"name": "colsample_bytree", "type": "continuous", "bounds": [0.5, 0.8]},
    ]


def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

