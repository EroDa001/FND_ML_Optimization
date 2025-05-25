import numpy as np
from mealpy import Optimizer, FloatVar
from sklearn.metrics import accuracy_score


class HideAndSeekOptimizer(Optimizer):
    def __init__(self, epoch=50, pop_size=20, seeker_ratio=0.3, p_replace=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.seeker_ratio = self.validator.check_float("seeker_ratio", seeker_ratio, (0, 1))
        self.p_replace = self.validator.check_float("p_replace", p_replace, (0, 1))
        self.sort_flag = True

    def initialize_variables(self):
        self.n_seekers = int(self.pop_size * self.seeker_ratio)
        self.n_hiders = self.pop_size - self.n_seekers
        self.space = self.problem.ub - self.problem.lb

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

    def evolve(self, epoch):
        epsilon = 1.0 - epoch / self.epoch

        for idx in range(self.n_hiders, self.pop_size):
            if self.generator.uniform() < self.p_replace:
                self.pop[idx] = self.generate_agent()

        best_hider = self.pop[0]
        for i in range(self.n_seekers):
            new_pos = self.pop[i].solution + epsilon * self.space * self.generator.uniform(-1, 1)
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(agent.target, self.pop[i].target, self.problem.minmax):
                self.pop[i] = agent

        for i in range(self.n_seekers, self.pop_size):
            direction = self.generator.normal(0, 1, self.problem.n_dims)
            new_pos = self.pop[i].solution + epsilon * self.space * direction
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(agent.target, self.pop[i].target, self.problem.minmax):
                self.pop[i] = agent


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

    def decode_solution(solution):
        params = {}
        for i, param in enumerate(space):
            if param["type"] == "continuous":
                params[param["name"]] = float(solution[i])
            elif param["type"] == "categorical":
                idx = int(round(solution[i]))
                idx = max(0, min(idx, len(cat_category_lists[i]) - 1))
                params[param["name"]] = cat_category_lists[i][idx]
        return params

    def fitness(solution):
        params = decode_solution(solution)
        model = module.create_model(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return -accuracy_score(y_val, preds)  # Minimization

    problem = {
        "obj_func": fitness,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "min"
    }

    model = HideAndSeekOptimizer(epoch=50, pop_size=20)
    g_best = model.solve(problem)

    best_params = decode_solution(g_best.solution)
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_val)
    score = accuracy_score(y_val, preds)

    if verbose:
        print("Best params (HideAndSeek):", {k: f"{v:.2e}" if isinstance(v, float) else v for k, v in best_params.items()})
        print(f"Best CV accuracy: {score:.4f}")

    return best_model, best_params, score
