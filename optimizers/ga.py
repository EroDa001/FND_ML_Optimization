from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sko.GA import GA

from experiments.config import GA_MAX_ITER, GA_MUTATION_RATE, GA_POP_SIZE


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
            elif p["type"] == "categorical":
                idx = int(round(solution[i]))
                name = p["name"]
                label = cat_encoders[cat_params.index(p)].inverse_transform([idx])[0]
                params[name] = label
            i += 1
        return params

    def fitness(solution):
        params = decode_solution(solution)
        model = model_module.create_model(params)
        score = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        ).mean()
        return score

    ga = GA(
        func=fitness,
        n_dim=dim,
        size_pop=GA_POP_SIZE,
        max_iter=GA_MAX_ITER,
        prob_mut=GA_MUTATION_RATE,
        lb=lb,
        ub=ub,
        precision=1e-4,
    )

    best_x, best_y = ga.run()

    best_params = decode_solution(best_x)
    best_model = model_module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print(f"Best params (GA): {best_params}")
        print(f"Best CV accuracy: {best_y.item():.4f}")
        # print(best_y)

    return best_model, best_params, best_y
