import numpy as np
from sklearn.metrics import f1_score
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

    # Bounds and categories for parameters
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

    lb, ub = np.array(lb), np.array(ub)

    def decode_solution(sol):
        params = {}
        for i, p in enumerate(space):
            if p["type"] == "continuous":
                params[p["name"]] = round(float(sol[i]), 4)
            else:
                idx = int(round(sol[i]))
                idx = np.clip(idx, 0, len(cat_category_lists[i]) - 1)
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

    # --- Soccer-Inspired Golden Ball Optimization ---
    NUM_TEAMS = 4  # Number of teams (subpopulations)
    TEAM_SIZE = GB_POP_SIZE // NUM_TEAMS  # Players per team
    TRANSFER_RATE = 0.2  # Probability of a player transferring
    TRAINING_INTENSITY = 0.5  # How much teams improve players locally

    # Initialize teams and players
    teams = []
    for _ in range(NUM_TEAMS):
        team = lb + np.random.rand(TEAM_SIZE, dim) * (ub - lb)
        teams.append(team)

    # Track the best player (Golden Ball winner)
    best_sol = None
    best_score = -np.inf

    for season in range(GB_MAX_ITER):
        # Evaluate all players in all teams
        all_players = np.concatenate(teams)
        all_fit = fitness(all_players)

        # Update Golden Ball (best overall player)
        current_best_idx = np.argmax(all_fit)
        if all_fit[current_best_idx] > best_score:
            best_score = all_fit[current_best_idx]
            best_sol = all_players[current_best_idx].copy()

        if verbose:
            print(f"Season {season+1}/{GB_MAX_ITER}, Best F1: {best_score:.4f}")

        # Team Training Phase (Local Search)
        for i in range(NUM_TEAMS):
            team_fit = all_fit[i*TEAM_SIZE : (i+1)*TEAM_SIZE]
            best_in_team_idx = np.argmax(team_fit)
            best_in_team = teams[i][best_in_team_idx]

            # Players learn from the best in their team
            for j in range(TEAM_SIZE):
                if np.random.rand() < TRAINING_INTENSITY:
                    r = np.random.rand(dim)
                    teams[i][j] += r * (best_in_team - teams[i][j])
                    teams[i][j] = np.clip(teams[i][j], lb, ub)

        # Transfer Phase (Global Exploration)
        if season < GB_MAX_ITER - 1:  # No transfers in the last season
            for i in range(NUM_TEAMS):
                for j in range(TEAM_SIZE):
                    if np.random.rand() < TRANSFER_RATE:
                        # Move to a random team (including current one)
                        target_team = np.random.randint(0, NUM_TEAMS)
                        teams[target_team][np.random.randint(0, TEAM_SIZE)] = teams[i][j].copy()

    # Return the Golden Ball winner (best model)
    best_params = decode_solution(best_sol)
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        formatted = {k: f"{v:.2e}" if isinstance(v, float) else v for k, v in best_params.items()}
        print(f"Golden Ball Winner (Best Params): {formatted}")
        print(f"Best F1-Score: {best_score:.4f}")

    return best_model, best_params, best_score