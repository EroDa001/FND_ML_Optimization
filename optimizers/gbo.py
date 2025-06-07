import numpy as np
from sklearn.base import clone
from experiments.config import GB_MAX_ITER ,GB_POP_SIZE 

def optimize(module, X_train, y_train, X_val, y_val, verbose=True):
    space = module.param_space()
    NT = 3  # number of groups
    NP = 2  # number of schedules per group
    max_seasons = 3
    optimal_found = False

    groups = []
    for i in range(NT):
        schedules = []
        for _ in range(NP):
            individual = []
            for param in space:
                if param['type'] == 'continuous':
                    individual.append(np.random.uniform(*param['bounds']))
                elif param['type'] == 'categorical':
                    individual.append(np.random.randint(len(param['categories'])))
            schedules.append(individual)
        training_fn = np.random.choice(['2opt', 'insertion', 'swap', 'ox'])
        groups.append({
            'schedules': schedules,
            'training': training_fn,
            'points': 0,
            'captain': None,
            'strength': None
        })

    def decode_solution(sol):
        decoded = {}
        for i, param in enumerate(space):
            if param['type'] == 'continuous':
                decoded[param['name']] = sol[i]
            elif param['type'] == 'categorical':
                decoded[param['name']] = param['categories'][int(round(sol[i]))]
        return decoded

    def evaluate(sol):
        params = decode_solution(sol)
        model = module.create_model(params)
        model.fit(X_train, y_train)
        return 1.0 - model.score(X_val, y_val)  # minimization

    def train(schedule, method):
        return schedule  # placeholder for training functions

    for season in range(max_seasons):
        print(f"Season {season + 1}/{max_seasons}")

        for group_index, group in enumerate(groups):
            scores = [evaluate(s) for s in group['schedules']]
            for idx, score in enumerate(scores):
                print(f"Group {group_index}, Player {idx}, Score: {1.0 - score:.4f}")
            best_idx = np.argmin(scores)
            group['captain'] = group['schedules'][best_idx]
            group['strength'] = np.mean(scores)

        for group in groups:
            group['schedules'] = [train(s, group['training']) for s in group['schedules']]

        for group in groups:
            group['schedules'].sort(key=lambda s: evaluate(s))

        for i in range(NT):
            for j in range(i + 1, NT):
                s1 = groups[i]['schedules'][0]
                s2 = groups[j]['schedules'][0]
                v1 = evaluate(s1)
                v2 = evaluate(s2)
                if abs(v1 - v2) < 1e-6:
                    groups[i]['points'] += 1
                    groups[j]['points'] += 1
                elif v1 < v2:
                    groups[i]['points'] += 3
                else:
                    groups[j]['points'] += 3

        if optimal_found:
            break

        groups.sort(key=lambda g: (g['points'], g['strength']))
        top = groups[:NT // 2]
        bottom = groups[NT // 2:]
        for i in range(len(top)):
            bottom[i]['schedules'][:NP//2], top[i]['schedules'][:NP//2] = top[i]['schedules'][:NP//2], bottom[i]['schedules'][:NP//2]
            bottom[i]['training'], top[i]['training'] = top[i]['training'], bottom[i]['training']

        for group in groups:
            group['points'] = 0

    all_best = sorted([(evaluate(g['captain']), g['captain']) for g in groups], key=lambda x: x[0])
    best_params = decode_solution(all_best[0][1])
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        print("Best params (GBO-style):", {k: f"{v:.2e}" if isinstance(v, float) else v for k, v in best_params.items()})
        print(f"Best CV error: {all_best[0][0]:.4f}")

    return best_model, best_params, 1.0 - all_best[0][0]
