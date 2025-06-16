import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.config import SARO_EPOCHS ,SARO_POP_SIZE

import numpy as np
from sklearn.metrics import f1_score
from mealpy.human_based import SARO
from mealpy import FloatVar

def optimize(module, X_train, y_train, X_val, y_val, verbose=True):
    space = module.param_space()
    dim = len(space)
    
    lb, ub = [], []
    cat_category_lists = []

    for param in space:
        if param["type"] == "continuous":
            lb.append(param["bounds"][0])
            ub.append(param["bounds"][1])
            cat_category_lists.append(None)
        elif param["type"] == "categorical":
            lb.append(0)
            ub.append(len(param["categories"]) - 1)
            cat_category_lists.append(param["categories"])

    def decode_solution(sol):
        params = {}
        for i, param in enumerate(space):
            if param["type"] == "continuous":
                params[param["name"]] = round(float(sol[i]), 4)
            else:
                idx = int(round(sol[i]))
                params[param["name"]] = cat_category_lists[i][idx]
        return params

    def fitness_func(sol):
        params = decode_solution(sol)
        model = module.create_model(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')  
        return -f1  


    bounds = FloatVar(lb=tuple(lb), ub=tuple(ub), name="hyperparams")
    problem = {
        "obj_func": fitness_func,
        "bounds": bounds,
        "minmax": "min"
    }

    model = SARO.DevSARO(
        epoch=SARO_EPOCHS, 
        pop_size=SARO_POP_SIZE, 
        se=0.9, 
        mu=7 
        )

    best = model.solve(problem)

    best_params = decode_solution(best.solution)
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(
            f"Best params (SARO): { {k: f'{v:.2e}' if isinstance(v, float) else v for k, v in best_params.items()} }"
        )
        print(f"Best CV F1-score: {-best.target.fitness:.4f}")

    return best_model, best_params, -best.target.fitness