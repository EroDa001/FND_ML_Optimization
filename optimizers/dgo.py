import numpy as np
from sklearn.metrics import f1_score 

from experiments.config import DGO_MAX_ITER , DGO_POP_SIZE , DGO_DICE_SIDES

def optimize(
    module,
    X_train,
    y_train,
    X_val,
    y_val,
    verbose=True,
    max_iter=DGO_MAX_ITER,
    pop_size=DGO_POP_SIZE,
    dice_sides=DGO_DICE_SIDES,  
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
            score = f1_score(y_val, y_pred, average='macro')  
            scores.append(score)
        return np.array(scores)


    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    best_pop, best_score = None, -np.inf

    for iteration in range(max_iter):
        fit = fitness(pop)
        if verbose:
            print(f"Iteration {iteration+1}/{max_iter}, Best fitness: {np.max(fit):.4f}")

        idx = np.argmax(fit)
        if fit[idx] > best_score:
            best_score = fit[idx]
            best_pop = pop[idx].copy()

        new_pop = pop.copy()

        for i in range(pop_size):

            dice_rolls = np.random.randint(1, dice_sides + 1, size=dim)

            
            step_sizes = (dice_rolls - 1) / (dice_sides - 1)

            
            direction = best_pop - pop[i]
            random_factor = np.random.rand(dim)

            
            new_position = pop[i] + direction * step_sizes * random_factor

            
            new_pop[i] = np.clip(new_position, lb, ub)

        pop = new_pop

    best_params = decode_solution(best_pop)
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(f"Best params (DGO): {best_params}")
        print(f"Best CV F1-score: {best_score:.4f}")

    return best_model, best_params, best_score
