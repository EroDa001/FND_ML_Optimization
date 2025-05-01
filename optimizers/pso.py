import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sko.PSO import PSO

from experiments.config import (CV_FOLDS, PSO_C1, PSO_C2, PSO_MAX_ITER,
                                PSO_POP_SIZE, PSO_W, RANDOM_SEED,
                                TRAIN_FRACTION)


def optimize(
    model_module,
    X_train,
    y_train,
    train_fraction=TRAIN_FRACTION,
    cv=CV_FOLDS,
    verbose=True,
):
    space = model_module.param_space()

    if train_fraction < 1.0:
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=train_fraction,
            stratify=y_train,
            random_state=RANDOM_SEED,
        )

    dim = len(space)
    lb = []
    ub = []

    cat_category_lists = []

    for p in space:
        if p["type"] == "continuous":
            lb.append(p["bounds"][0])
            ub.append(p["bounds"][1])
            cat_category_lists.append(None)
        elif p["type"] == "categorical":
            lb.append(0)
            ub.append(len(p["categories"]) - 1)
            cat_category_lists.append(p["categories"])

    def decode_solution(solution):
        params = {}
        for i, p in enumerate(space):
            if p["type"] == "continuous":
                params[p["name"]] = solution[i]
            elif p["type"] == "categorical":
                idx = int(round(solution[i]))
                params[p["name"]] = cat_category_lists[i][idx]
        return params

    def fitness(solution):
        params = decode_solution(solution)
        model = model_module.create_model(params)
        score = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        ).mean()
        return -score  # PSO minimizes

    pso = PSO(
        func=fitness,
        dim=dim,
        pop=PSO_POP_SIZE,
        max_iter=PSO_MAX_ITER,
        lb=lb,
        ub=ub,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
    )

    best_x, best_y = pso.run()
    best_params = decode_solution(best_x)
    best_model = model_module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(
            f"Best params (PSO): { {k: f"{v:.2e}" if isinstance(v, np.float64) else v for k, v in best_params.items()} }"
        )
        print(f"Best CV accuracy: {-best_y.item():.4f}")

    return best_model, best_params, -best_y
