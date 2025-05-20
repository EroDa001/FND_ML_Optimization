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
    MIN_TEAM_SIZE = max(2, TEAM_SIZE // 2)  # Minimum team size after relegation
    TRANSFER_RATE = 0.4  # Base probability of a player transferring
    BASE_TRAINING_INTENSITY = 0.7  # Base training intensity
    
    # Initialize teams and players
    teams = []
    for _ in range(NUM_TEAMS):
        team = lb + np.random.rand(TEAM_SIZE, dim) * (ub - lb)
        teams.append(team)

    # Track the best player (Golden Ball winner)
    best_sol = None
    best_score = -np.inf
    stagnation_count = 0
    prev_best_score = -np.inf

    for season in range(GB_MAX_ITER):
        # Dynamic parameters that change each season
        dynamic_transfer_rate = TRANSFER_RATE * (1 + season/GB_MAX_ITER)  # Increases over time
        training_intensity = BASE_TRAINING_INTENSITY * (1 - season/GB_MAX_ITER)  # Decreases over time
        
        # Evaluate all players in all teams
        all_players = np.concatenate(teams)
        all_fit = fitness(all_players)

        # Update Golden Ball (best overall player)
        current_best_idx = np.argmax(all_fit)
        current_best_score = all_fit[current_best_idx]
        
        # Check for stagnation
        if current_best_score <= prev_best_score:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best_score = current_best_score

        if current_best_score > best_score:
            best_score = current_best_score
            best_sol = all_players[current_best_idx].copy()
            stagnation_count = 0  # Reset if we find improvement

        if verbose:
            print(f"Season {season+1}/{GB_MAX_ITER}, Best F1: {best_score:.4f}, Stagnation: {stagnation_count}")

        # Team Ranking and Relegation (dynamically adjust team sizes)
        team_performance = []
        for i in range(NUM_TEAMS):
            team_performance.append(np.mean(all_fit[i*TEAM_SIZE:(i+1)*TEAM_SIZE]))
        
        # Sort teams by performance (worst first)
        ranked_teams = np.argsort(team_performance)
        
        # Relegate players from worst teams to promote exploration
        if stagnation_count > 3:  # If stuck for 3 seasons
            for i in range(NUM_TEAMS // 2):  # Bottom half teams
                team_idx = ranked_teams[i]
                # Replace worst players with random new ones
                team_fit = all_fit[team_idx*TEAM_SIZE:(team_idx+1)*TEAM_SIZE]
                worst_player_idx = np.argmin(team_fit)
                teams[team_idx][worst_player_idx] = lb + np.random.rand(dim) * (ub - lb)

        # Team Training Phase (Local Search)
        for i in range(NUM_TEAMS):
            team_fit = all_fit[i*TEAM_SIZE : (i+1)*TEAM_SIZE]
            best_in_team_idx = np.argmax(team_fit)
            best_in_team = teams[i][best_in_team_idx]

            for j in range(TEAM_SIZE):
                r = np.random.rand(dim)
                # Dynamic training intensity
                if np.random.rand() < training_intensity:
                    # 40% chance to learn from global best, 60% from team best
                    if np.random.rand() < 0.4:
                        teams[i][j] += r * (best_sol - teams[i][j])
                    else:
                        teams[i][j] += r * (best_in_team - teams[i][j])
                    # Clip to bounds
                    teams[i][j] = np.clip(teams[i][j], lb, ub)
                
                # Adaptive mutation (more aggressive when stagnating)
                mutation_prob = 0.1 + 0.1 * (stagnation_count / 5)  # Increases with stagnation
                if np.random.rand() < mutation_prob:
                    mutation_size = 0.3 * (1 + stagnation_count / 5)  # Increases with stagnation
                    teams[i][j] += np.random.normal(0, mutation_size, dim) * (ub - lb)
                    teams[i][j] = np.clip(teams[i][j], lb, ub)

        # Transfer Phase (Global Exploration)
        if season < GB_MAX_ITER - 1:  # No transfers in the last season
            for i in range(NUM_TEAMS):
                for j in range(TEAM_SIZE):
                    # Dynamic transfer rate increases with stagnation
                    current_transfer_rate = min(0.8, dynamic_transfer_rate * (1 + stagnation_count / 5))
                    if np.random.rand() < current_transfer_rate:
                        # Prefer transferring to better teams
                        team_weights = np.array(team_performance) - min(team_performance)
                        if team_weights.sum() > 0:
                            team_weights = team_weights / team_weights.sum()
                            target_team = np.random.choice(NUM_TEAMS, p=team_weights)
                        else:
                            target_team = np.random.randint(0, NUM_TEAMS)
                        # Replace a random player (could be from any team)
                        replace_team = np.random.randint(0, NUM_TEAMS)
                        replace_player = np.random.randint(0, TEAM_SIZE)
                        teams[replace_team][replace_player] = teams[i][j].copy()

    # Return the Golden Ball winner (best model)
    best_params = decode_solution(best_sol)
    best_model = module.create_model(best_params)
    best_model.fit(X_train, y_train)

    if verbose:
        formatted = {k: f"{v:.2e}" if isinstance(v, float) else v for k, v in best_params.items()}
        print(f"Golden Ball Winner (Best Params): {formatted}")
        print(f"Best F1-Score: {best_score:.4f}")

    return best_model, best_params, best_score