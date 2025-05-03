import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from experiments.config import (CV_FOLDS, RANDOM_SEED, TRAIN_FRACTION,
                                TWO_MAX_ITER, TWO_POP_SIZE)


def optimize(
    model_module,
    X_train,
    y_train,
    cv=CV_FOLDS,
    train_fraction=TRAIN_FRACTION,
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

    lb, ub = [], []
    cat_category_lists = []

    for p in space:
        if p["type"] == "continuous":
            lb.append(p["bounds"][0])
            ub.append(p["bounds"][1])
            cat_category_lists.append(None)
        else:
            lb.append(0)
            ub.append(len(p["categories"]) - 1)
            cat_category_lists.append(p["categories"])

    lb = np.array(lb)
    ub = np.array(ub)

    def decode_solution(sol):
        params = {}
        for i, p in enumerate(space):
            if p["type"] == "continuous":
                params[p["name"]] = float(sol[i])
            else:
                idx = int(round(sol[i]))
                params[p["name"]] = cat_category_lists[i][idx]
        return params

    def fitness_vals(pop):
        scores = []
        for sol in pop:
            params = decode_solution(sol)
            model = model_module.create_model(params)
            acc = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1
            ).mean()
            scores.append(acc)
        return np.array(scores)

    pop = lb + np.random.rand(TWO_POP_SIZE, dim) * (ub - lb)
    best_pop, best_score = None, -np.inf

    for it in range(TWO_MAX_ITER):
        fit = fitness_vals(pop)
        weights = fit / (fit.sum() + 1e-16)

        idx = np.argmax(fit)
        if fit[idx] > best_score:
            best_score = fit[idx]
            best_pop = pop[idx].copy()

        new_pop = pop.copy()
        for i in range(TWO_POP_SIZE):
            force = np.zeros(dim)
            for j in range(TWO_POP_SIZE):
                if i == j:
                    continue
                diff = pop[j] - pop[i]
                force += np.random.rand(dim) * weights[j] * diff
            new_pop[i] = pop[i] + force
        pop = np.clip(new_pop, lb, ub)

        # if verbose and (it + 1) % 10 == 0:
        #     print(f"[TWO] Iter {it+1}/{TWO_MAX_ITER} — best CV acc: {best_score:.4f}")

    best_params = decode_solution(best_pop)
    best_model = model_module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(
            f"Best params (TOW): { {k: f"{v:.2e}" if isinstance(v, np.float64) else v for k, v in best_params.items()} }"
        )
        print(f"Best CV accuracy: {best_score:.4f}")

    return best_model, best_params, best_score
