import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm
from matplotlib import pyplot
from scipy.optimize import least_squares
import pickle
from tqdm import tqdm

data = pd.read_csv("train.csv")
data["win"] = data["team_score"] > data["opponent_team_score"]
data["diff"] = data["team_score"] - data["opponent_team_score"]

teams = data["team"].unique().tolist()
data["team_id"] = data["team"].apply(lambda x: teams.index(x))
data["opponent_team_id"] = data.groupby("game_id")["team_id"].transform("sum") - data["team_id"]

def fit_opr_dpr(games, num_teams):
    """
    Solve for the Offensive Power Rating (OPR) and Defensive Power Rating (DPR) for each team.
    
    :param games: List of tuples (team_A, team_B, score_A, score_B)
    :param num_teams: Total number of teams
    :return: OPR and DPR arrays
    """
    params = np.zeros(2 * num_teams)
    
    def loss_function(x):
        opr = x[:num_teams]
        dpr = x[num_teams:]
        residuals = []
        
        for team_A, team_B, score_A, score_B in games:
            pred_A = opr[team_A] - dpr[team_B]
            pred_B = opr[team_B] - dpr[team_A]
            residuals.append(pred_A - score_A)
            residuals.append(pred_B - score_B)
        
        return np.array(residuals)
    
    return least_squares(loss_function, params)

labels = data.columns[3:18]
for label in tqdm(labels):
    data["score"] = data[label]
    data["opponent_score"] = data.groupby("game_id")["score"].transform("sum") - data["score"]
    
    games = data[["team_id", "opponent_team_id", "score", "opponent_score"]].to_numpy()
    result = fit_opr_dpr(games, len(teams))
    
    with open(f"result-{label}.pkl", "wb") as f:
        pickle.dump(result, f)