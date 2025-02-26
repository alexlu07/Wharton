import numpy as np
import pandas as pd
import pickle
from scipy.linalg import lstsq

data = pd.read_csv("train.csv")
data["win"] = data["team_score"] > data["opponent_team_score"]
data["diff"] = data["team_score"] - data["opponent_team_score"]

teams = data["team"].unique().tolist()
data["team_id"] = data["team"].apply(lambda x: teams.index(x))
data["opponent_team_id"] = data.groupby("game_id")["team_id"].transform("sum") - data["team_id"]

def least_squares(df, num_teams):
    """
    Computes Offensive Power Rating (OPR) and Defensive Power Rating (DPR) 
    for multiple score columns using least squares.

    Parameters:
    df : DataFrame
        A dataframe with the first two columns as teamA and teamB ids
    num_teams : int
        Total number of unique teams

    Returns:
    oprs : (num_teams, num_scores) array
        Offensive ratings for each team across score columns.
    dprs : (num_teams, num_scores) array
        Defensive ratings for each team across score columns.
    """

    # Construct one-hot coefficient matrix A (size G x 2T)
    A = np.zeros((df.shape[0], 2 * num_teams))

    # Set values in A matrix for OPR and DPR equations
    A[np.arange(A.shape[0]), df["team_id"]] = 1   # OPR_A
    A[np.arange(A.shape[0]), num_teams + df["opponent_team_id"]] = -1  # -DPR_B

    # Construct B matrix (size G x num_scores)
    b = df.iloc[:, 2:].values

    # Solve using least squares with QR (gelsy)
    x, residues, _, _ = lstsq(A, b, lapack_driver="gelsy")

    # Split OPR and DPR
    oprs = x[:num_teams]  # OPR values
    dprs = x[num_teams:]  # DPR values

    return x, oprs, dprs, residues

games = data[["team_id", "opponent_team_id", *data.columns[3:19]]]
result = least_squares(games, len(teams))

with open(f"result.pkl", "wb") as f:
    pickle.dump(result, f)
