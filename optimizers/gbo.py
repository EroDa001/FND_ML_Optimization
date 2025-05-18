import numpy as np
from experiments.config import GB_MAX_ITER, GB_POP_SIZE

def optimize(
    module,
    X_train,
    y_train,
    X_val,
    y_val,
    verbose=True,
):
    space = module.param_space()
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
                params[p["name"]] = round(float(sol[i]), 4)
            else:
                idx = int(round(sol[i]))
                params[p["name"]] = cat_category_lists[i][idx]
        return params

def fitness(pop):
    scores = []
    for sol in pop:
        params = decode_solution(sol)
        model = module.create_model(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred)
        scores.append(score)
    return np.array(scores)

    # Initialize population
    pop = lb + np.random.rand(GB_POP_SIZE, dim) * (ub - lb)
    best_pop, best_score = None, -np.inf

    for _ in range(GB_MAX_ITER):
        fit = fitness(pop)
        idx = np.argmax(fit)
        if fit[idx] > best_score:
            best_score = fit[idx]
            best_pop = pop[idx].copy()

        # Update population
        new_pop = pop.copy()
        for i in range(GB_POP_SIZE):
            if i == idx:
                continue
            direction = best_pop - pop[i]
            new_pop[i] = pop[i] + np.random.rand(dim) * direction
        pop = np.clip(new_pop, lb, ub)

    best_params = decode_solution(best_pop)
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(
            f"Best params (GB): { {k: f'{v:.2e}' if isinstance(v, float) else v for k, v in best_params.items()} }"
        )
        print(f"Best CV accuracy: {best_score:.4f}")

    return best_model, best_params, best_score
