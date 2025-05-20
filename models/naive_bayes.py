from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def create_model(params):
    model = MultinomialNB(alpha=params["alpha"])
    return Pipeline([("estimator", model)])


def default_params():
    return {"alpha": 1e-3}


def param_space():
    return [
        {"name": "alpha", "type": "continuous", "bounds": [1e-6, 1.0]},
    ]


def evaluate(model, X_val, y_val):
    return accuracy_score(y_val, model.predict(X_val))
