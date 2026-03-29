"""
Game definitions and Nash equilibrium computation.

Convention:
  payoff_A[i, j] = P1's payoff when P1 plays action i, P2 plays action j
  payoff_B[j, i] = P2's payoff when P2 plays action j, P1 plays action i
  For zero-sum games: payoff_B = -payoff_A.T
"""

import numpy as np
from scipy.optimize import linprog


GAMES: dict = {
    "Rock Paper Scissors": {
        "actions_p1": ["Rock", "Paper", "Scissors"],
        "actions_p2": ["Rock", "Paper", "Scissors"],
        "payoff_A": np.array([
            [ 0, -1,  1],
            [ 1,  0, -1],
            [-1,  1,  0],
        ], dtype=float),
        "zero_sum": True,
        "description": (
            "Classic 3-action zero-sum game. "
            "Unique Nash equilibrium: uniform (1/3, 1/3, 1/3) for both players. "
            "Game value = 0."
        ),
    },
    "Matching Pennies": {
        "actions_p1": ["Heads", "Tails"],
        "actions_p2": ["Heads", "Tails"],
        "payoff_A": np.array([
            [ 1, -1],
            [-1,  1],
        ], dtype=float),
        "zero_sum": True,
        "description": (
            "2-action zero-sum game. P1 wins if both show same side; P2 wins otherwise. "
            "Nash equilibrium: uniform (1/2, 1/2) for both. Game value = 0."
        ),
    },
    "Biased RPS": {
        "actions_p1": ["Rock", "Paper", "Scissors"],
        "actions_p2": ["Rock", "Paper", "Scissors"],
        "payoff_A": np.array([
            [ 0, -1,  2],
            [ 1,  0, -1],
            [-2,  1,  0],
        ], dtype=float),
        "zero_sum": True,
        "description": (
            "RPS variant where Scissors beating Rock pays 2 (and losing costs 2). "
            "Nash equilibrium shifts away from uniform — algorithms should track this."
        ),
    },
    "Prisoner's Dilemma": {
        "actions_p1": ["Cooperate", "Defect"],
        "actions_p2": ["Cooperate", "Defect"],
        "payoff_A": np.array([
            [3, 0],
            [5, 1],
        ], dtype=float),
        "payoff_B": np.array([
            [3, 5],
            [0, 1],
        ], dtype=float),
        "zero_sum": False,
        "description": (
            "Classic general-sum dilemma. "
            "Dominant strategy: Defect. Nash = (Defect, Defect) even though (C,C) is Pareto-better. "
            "Not zero-sum — regret minimizers may not converge to Nash here."
        ),
    },
    "Battle of the Sexes": {
        "actions_p1": ["Opera", "Football"],
        "actions_p2": ["Opera", "Football"],
        "payoff_A": np.array([
            [2, 0],
            [0, 1],
        ], dtype=float),
        "payoff_B": np.array([
            [1, 0],
            [0, 2],
        ], dtype=float),
        "zero_sum": False,
        "description": (
            "General-sum coordination game with two pure Nash equilibria: (Opera, Opera) and "
            "(Football, Football), plus a mixed Nash. Algorithms may cycle or converge to mixed NE."
        ),
    },
    "Custom": {
        "actions_p1": ["A", "B"],
        "actions_p2": ["A", "B"],
        "payoff_A": np.array([
            [1.0, -1.0],
            [-1.0, 1.0],
        ], dtype=float),
        "zero_sum": True,
        "description": "Edit the payoff matrix below.",
    },
}


def get_payoff_B(game: dict) -> np.ndarray:
    """Return P2's payoff matrix (rows=P2 actions, cols=P1 actions)."""
    if "payoff_B" in game:
        return game["payoff_B"].copy()
    return -game["payoff_A"].T


def compute_nash_zero_sum(payoff_A: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve a zero-sum game via linear programming.

    Returns (nash_row, nash_col, game_value).

    Row player LP:
        max  v
        s.t. (A^T x)_j >= v  for all j      (x is not dominated column-wise)
             sum(x) = 1,  x >= 0
    """
    n, m = payoff_A.shape

    # --- Row player ---
    # Variables: [x_0, ..., x_{n-1}, v]
    c_row = np.zeros(n + 1)
    c_row[-1] = -1.0  # minimize -v

    # -A^T x + v <= 0  →  A_ub @ z <= b_ub
    A_ub_row = np.hstack([-payoff_A.T, np.ones((m, 1))])
    b_ub_row = np.zeros(m)

    A_eq_row = np.zeros((1, n + 1))
    A_eq_row[0, :n] = 1.0
    b_eq_row = np.array([1.0])

    bounds_row = [(0.0, None)] * n + [(None, None)]

    res_row = linprog(
        c_row, A_ub=A_ub_row, b_ub=b_ub_row,
        A_eq=A_eq_row, b_eq=b_eq_row,
        bounds=bounds_row, method="highs",
    )

    # --- Column player ---
    # Variables: [y_0, ..., y_{m-1}, w]
    c_col = np.zeros(m + 1)
    c_col[-1] = 1.0  # minimize w (= max_x min_y x^T A y)

    # A y - w*1 <= 0  →  A_ub @ z <= b_ub
    A_ub_col = np.hstack([payoff_A, -np.ones((n, 1))])
    b_ub_col = np.zeros(n)

    A_eq_col = np.zeros((1, m + 1))
    A_eq_col[0, :m] = 1.0
    b_eq_col = np.array([1.0])

    bounds_col = [(0.0, None)] * m + [(None, None)]

    res_col = linprog(
        c_col, A_ub=A_ub_col, b_ub=b_ub_col,
        A_eq=A_eq_col, b_eq=b_eq_col,
        bounds=bounds_col, method="highs",
    )

    if res_row.success and res_col.success:
        nash_row = res_row.x[:n]
        nash_col = res_col.x[:m]
        game_value = float(-res_row.fun)
        return nash_row, nash_col, game_value

    raise ValueError(f"LP failed: row={res_row.message}, col={res_col.message}")


def exploitability(
    p1_mixed: np.ndarray,
    p2_mixed: np.ndarray,
    payoff_A: np.ndarray,
    payoff_B: np.ndarray,
) -> tuple[float, float]:
    """
    Return (eps_1, eps_2): how much each player could gain by deviating.

    eps_1 = max_i A[i,:] @ p2_mixed - p1_mixed @ A @ p2_mixed
    eps_2 = max_j B[j,:] @ p1_mixed - p2_mixed @ B @ p1_mixed
    """
    p1_ev = p1_mixed @ payoff_A @ p2_mixed
    best_p1 = float((payoff_A @ p2_mixed).max())
    eps_1 = max(0.0, best_p1 - p1_ev)

    p2_ev = p2_mixed @ payoff_B @ p1_mixed
    best_p2 = float((payoff_B @ p1_mixed).max())
    eps_2 = max(0.0, best_p2 - p2_ev)

    return eps_1, eps_2
