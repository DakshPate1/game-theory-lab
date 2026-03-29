"""
Simulation runner for two-player games.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from games import get_payoff_B


@dataclass
class SimulationResult:
    # Per-round records
    p1_mixed: list[np.ndarray] = field(default_factory=list)  # shape (T, n1)
    p2_mixed: list[np.ndarray] = field(default_factory=list)  # shape (T, n2)
    p1_actions: list[int] = field(default_factory=list)
    p2_actions: list[int] = field(default_factory=list)
    p1_payoffs: list[float] = field(default_factory=list)
    p2_payoffs: list[float] = field(default_factory=list)

    # Regret tracking (cumulative counterfactual payoffs per action)
    p1_cf_payoffs: list[np.ndarray] = field(default_factory=list)  # shape (T, n1)
    p2_cf_payoffs: list[np.ndarray] = field(default_factory=list)

    # Exploitability (Nash gap) — only filled for zero-sum games
    p1_exploit: list[float] = field(default_factory=list)
    p2_exploit: list[float] = field(default_factory=list)

    # Time-average mixed strategies
    @property
    def p1_avg_mixed(self) -> np.ndarray:
        return np.mean(self.p1_mixed, axis=0)

    @property
    def p2_avg_mixed(self) -> np.ndarray:
        return np.mean(self.p2_mixed, axis=0)

    @property
    def p1_regret(self) -> np.ndarray:
        """External regret at each round t (array of length T)."""
        cf = np.array(self.p1_cf_payoffs)   # (T, n1)
        actual = np.array(self.p1_payoffs)   # (T,)
        cum_cf = cf.cumsum(axis=0)           # (T, n1)
        cum_actual = actual.cumsum()         # (T,)
        T_range = np.arange(1, len(actual) + 1)
        return (cum_cf.max(axis=1) - cum_actual) / T_range

    @property
    def p2_regret(self) -> np.ndarray:
        cf = np.array(self.p2_cf_payoffs)
        actual = np.array(self.p2_payoffs)
        cum_cf = cf.cumsum(axis=0)
        cum_actual = actual.cumsum()
        T_range = np.arange(1, len(actual) + 1)
        return (cum_cf.max(axis=1) - cum_actual) / T_range

    @property
    def p1_cum_payoff(self) -> np.ndarray:
        return np.cumsum(self.p1_payoffs)

    @property
    def p2_cum_payoff(self) -> np.ndarray:
        return np.cumsum(self.p2_payoffs)


def run_simulation(
    game: dict,
    p1_strategy,
    p2_strategy,
    n_rounds: int = 1000,
    seed: int | None = 42,
    exploit_every: int = 50,
    adversarial_p2: bool = False,
    adversarial_p1: bool = False,
) -> SimulationResult:
    """
    Run n_rounds of the game, recording history.

    exploit_every:   how often to compute exploitability (set 0 to skip).
    adversarial_p2:  if True, P2 sees P1's mixed strategy before choosing (Stackelberg).
    adversarial_p1:  if True, P1 sees P2's mixed strategy before choosing.
    """
    rng = np.random.default_rng(seed)
    payoff_A = game["payoff_A"]            # (n1, n2) — P1's payoff
    payoff_B = get_payoff_B(game)          # (n2, n1) — P2's payoff
    n1 = payoff_A.shape[0]
    n2 = payoff_A.shape[1]
    zero_sum = game.get("zero_sum", True)

    result = SimulationResult()

    for t in range(n_rounds):
        # --- compute mixed strategies (with optional adversarial observation) ---
        if adversarial_p2:
            # P1 commits first; P2 sees P1's strategy before choosing
            x = p1_strategy.get_mixed_strategy(payoff_A)
            p2_strategy.observe_opponent_strategy(x)
            y = p2_strategy.get_mixed_strategy(payoff_B)
        elif adversarial_p1:
            # P2 commits first; P1 sees P2's strategy before choosing
            y = p2_strategy.get_mixed_strategy(payoff_B)
            p1_strategy.observe_opponent_strategy(y)
            x = p1_strategy.get_mixed_strategy(payoff_A)
        else:
            # Standard simultaneous play
            x = p1_strategy.get_mixed_strategy(payoff_A)
            y = p2_strategy.get_mixed_strategy(payoff_B)

        a1 = int(rng.choice(n1, p=x))
        a2 = int(rng.choice(n2, p=y))

        payoff1 = float(payoff_A[a1, a2])
        payoff2 = float(payoff_B[a2, a1])

        # counterfactual payoff vectors
        cf1 = payoff_A[:, a2].copy()   # what P1 would have gotten for each action
        cf2 = payoff_B[:, a1].copy()

        # --- update strategies ---
        p1_strategy.update(a1, a2, cf1)
        p2_strategy.update(a2, a1, cf2)

        # --- record ---
        result.p1_mixed.append(x)
        result.p2_mixed.append(y)
        result.p1_actions.append(a1)
        result.p2_actions.append(a2)
        result.p1_payoffs.append(payoff1)
        result.p2_payoffs.append(payoff2)
        result.p1_cf_payoffs.append(cf1)
        result.p2_cf_payoffs.append(cf2)

        # exploitability (Nash gap)
        if exploit_every > 0 and (t % exploit_every == 0 or t == n_rounds - 1):
            from games import exploitability
            avg_x = np.mean(result.p1_mixed, axis=0)
            avg_y = np.mean(result.p2_mixed, axis=0)
            eps1, eps2 = exploitability(avg_x, avg_y, payoff_A, payoff_B)
            result.p1_exploit.append(eps1)
            result.p2_exploit.append(eps2)

    return result
