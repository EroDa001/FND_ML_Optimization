import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sko.SA import SA

from experiments.config import SA_L, SA_MAX_ITER, SA_T_MAX, SA_T_MIN


def optimize(model_module, X_train, y_train, cv=5, verbose=True):
    space = model_module.param_space()

    cont_params = [p for p in space if p["type"] == "continuous"]
    cat_params = [p for p in space if p["type"] == "categorical"]

    cat_encoders = []
    for p in cat_params:
        encoder = LabelEncoder()
        encoder.fit(p["categories"])
        cat_encoders.append(encoder)

    dim = len(space)

    lb = []
    ub = []
    for p in space:
        if p["type"] == "continuous":
            lb.append(p["bounds"][0])
            ub.append(p["bounds"][1])
        elif p["type"] == "categorical":
            lb.append(0)
            ub.append(len(p["categories"]) - 1)

    def decode_solution(solution):
        params = {}
        i = 0
        for p in space:
            if p["type"] == "continuous":
                params[p["name"]] = solution[i]
            else:
                idx = int(round(solution[i]))
                label = cat_encoders[cat_params.index(p)].inverse_transform([idx])[0]
                params[p["name"]] = label
            i += 1
        return params

    def objective(sol):
        params = decode_solution(sol)
        model = model_module.create_model(params)
        score = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy"
        ).mean()
        return -score

    sa = SA(
        func=objective,
        x0=np.array([(lb[i] + ub[i]) / 2 for i in range(dim)]),  # initial point
        T_max=SA_T_MAX,
        T_min=SA_T_MIN,
        L=SA_L,
        max_iter=SA_MAX_ITER,
        lb=lb,
        ub=ub,
    )

    best_x, best_y = sa.run()
    best_params = decode_solution(best_x)
    best_model = model_module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(f"Best params (SA): {best_params}")
        print(f"Best CV accuracy: {-best_y.item():.4f}")

    return best_model, best_params, -best_y
