from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier

from experiments.config import RANDOM_SEED


def create_model(params):
    return CascadeForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_layers=int(params["max_layers"]),
        n_trees=int(params["n_trees"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def default_params():
    return {
        "n_estimators": 1,             
        "max_layers": 5,               
        "n_trees": 50,                 
        "min_samples_leaf": 5,         
    }

def param_space():
    return [
        {
            "name": "n_estimators",
            "type": "categorical",
            "categories": [1, 2],
        },
        {
            "name": "max_layers",
            "type": "categorical",
            "categories": list(range(3, 8)),  
        },
        {
            "name": "n_trees",
            "type": "categorical",
            "categories": [50, 100, 150],     
        },
        {
            "name": "min_samples_leaf",
            "type": "categorical",
            "categories": [2, 5, 10],         
        },
    ]



def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)



