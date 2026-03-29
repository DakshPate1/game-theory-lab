"""
Strategy / algorithm implementations for two-player games.

All strategies share the same interface:

    class SomeStrategy:
        name: str
        params: dict          # hyperparameter metadata for UI
        description: str
        theory: str           # LaTeX / markdown explanation

        def reset(self, n_actions: int, **kwargs) -> None
        def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray
        def update(self, my_action: int, opp_action: int,
                   payoff_vec: np.ndarray) -> None

    payoff_matrix[i, j] = my payoff when I play i and opponent plays j
    payoff_vec[i]        = payoff I would have gotten for action i this round
                         = payoff_matrix[i, opp_action]
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of vector v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.where(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
    if len(rho_candidates) == 0:
        return np.ones(n) / n
    rho = rho_candidates[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    w = np.exp(shifted)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str = ""
    description: str = ""
    theory: str = ""
    params: dict = field(default_factory=dict)   # {param_name: {default, min, max, step, help}}

    def reset(self, n_actions: int, **kwargs: Any) -> None:
        self.n_actions = n_actions
        self._opp_mixed: np.ndarray | None = None  # set by adversarial mode

    def observe_opponent_strategy(self, opp_mixed: np.ndarray) -> None:
        """Called before get_mixed_strategy in adversarial mode — opponent's current mixed strategy."""
        self._opp_mixed = opp_mixed

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        pass  # stateless by default


# ---------------------------------------------------------------------------
# Uniform Random
# ---------------------------------------------------------------------------

class Uniform(Strategy):
    name = "Uniform Random"
    description = "Always plays each action with equal probability."
    theory = (
        "No learning. Equivalent to Hedge with η→0 or FTRL with R→∞. "
        "Useful as a baseline — any adaptive algorithm should outperform it against a fixed opponent."
    )
    params = {}

    def reset(self, n_actions: int, **kwargs: Any) -> None:
        super().reset(n_actions)

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        return np.ones(self.n_actions) / self.n_actions


# ---------------------------------------------------------------------------
# Fixed Pure Strategy
# ---------------------------------------------------------------------------

class PureStrategy(Strategy):
    name = "Fixed Pure"
    description = "Always plays the same action."
    theory = (
        "Deterministic non-adaptive strategy. Against a learner, this is an easy target: "
        "the learner will quickly identify and exploit the pure best response."
    )
    params = {
        "action": {"default": 0, "min": 0, "max": 10, "step": 1,
                   "help": "Action index to always play (0-indexed)."},
    }

    def reset(self, n_actions: int, action: int = 0, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.action = min(action, n_actions - 1)

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        x = np.zeros(self.n_actions)
        x[self.action] = 1.0
        return x


# ---------------------------------------------------------------------------
# Hedge (Multiplicative Weights Update)
# ---------------------------------------------------------------------------

class Hedge(Strategy):
    name = "Hedge (MWU)"
    description = (
        "Exponential weights algorithm. Multiplicatively updates action weights "
        "proportional to observed payoffs."
    )
    theory = (
        r"**Update rule:** $w_i^{t+1} = w_i^t \cdot e^{\eta \cdot \ell_i^t}$, "
        r"then normalize: $x^{t+1} = w^{t+1} / \|w^{t+1}\|_1$."
        "\n\n"
        r"**Regret bound:** $R_T \leq \frac{\ln n}{\eta} + \eta T$. "
        r"Optimal $\eta = \sqrt{\ln n / T}$ gives $R_T = O(\sqrt{T \ln n})$."
        "\n\n"
        "**Equivalence:** Hedge = OMD with entropic mirror map = "
        "FTRL with negative-entropy regularizer R(x) = Σ xᵢ ln xᵢ."
    )
    params = {
        "eta": {"default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01,
                "help": "Learning rate η. Larger → faster but noisier adaptation."},
    }

    def reset(self, n_actions: int, eta: float = 0.1, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.eta = eta
        self.log_weights = np.zeros(n_actions)  # work in log space for stability

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        return _softmax(self.log_weights)

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self.log_weights += self.eta * payoff_vec


# ---------------------------------------------------------------------------
# FTRL — Follow the Regularized Leader
# ---------------------------------------------------------------------------

class FTRL_Entropy(Strategy):
    name = "FTRL (Entropy)"
    description = (
        "Follow the Regularized Leader with negative-entropy regularizer. "
        "Mathematically identical to Hedge."
    )
    theory = (
        r"**FTRL update:** $x^{t+1} = \arg\min_{x \in \Delta} "
        r"\left[ \eta \langle \sum_{s \leq t} \ell^s, x \rangle + R(x) \right]$"
        "\n\n"
        r"With $R(x) = \sum_i x_i \ln x_i$ (negative entropy), "
        r"the solution is $x_i \propto \exp\!\left(\eta \sum_{s} \ell_i^s\right)$, "
        "which is exactly Hedge."
        "\n\n"
        "**Key insight:** FTRL with entropy regularizer and OMD with entropic mirror map "
        "are two different algorithmic derivations of the **same update rule**."
    )
    params = {
        "eta": {"default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01,
                "help": "Learning rate η."},
    }

    def reset(self, n_actions: int, eta: float = 0.1, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.eta = eta
        self.cumulative_payoff = np.zeros(n_actions)

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        return _softmax(self.eta * self.cumulative_payoff)

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self.cumulative_payoff += payoff_vec


class FTRL_L2(Strategy):
    name = "FTRL (L2 / Euclidean)"
    description = (
        "Follow the Regularized Leader with squared-norm regularizer. "
        "Equivalent to projected online gradient ascent on the simplex."
    )
    theory = (
        r"**Regularizer:** $R(x) = \frac{1}{2}\|x\|_2^2$"
        "\n\n"
        r"**Solution:** $x^{t+1} = \Pi_\Delta\!\left(\eta \sum_{s \leq t} \ell^s\right)$, "
        "where Π_Δ is the Euclidean projection onto the simplex."
        "\n\n"
        r"**Regret bound:** $R_T = O(\sqrt{T})$ (same asymptotic as Hedge, "
        "but different constants and behavior — L2 can assign zero probability to actions)."
    )
    params = {
        "eta": {"default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01,
                "help": "Learning rate η."},
    }

    def reset(self, n_actions: int, eta: float = 0.1, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.eta = eta
        self.cumulative_payoff = np.zeros(n_actions)

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        return _project_simplex(self.eta * self.cumulative_payoff)

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self.cumulative_payoff += payoff_vec


# ---------------------------------------------------------------------------
# OMD — Online Mirror Descent
# ---------------------------------------------------------------------------

class OMD_Entropic(Strategy):
    name = "OMD (Entropic)"
    description = (
        "Online Mirror Descent with the entropic mirror map. "
        "Equivalent to Hedge — shown here for the alternative algorithmic derivation."
    )
    theory = (
        r"**Mirror map:** $\Phi(x) = \sum_i x_i \ln x_i$ (negative entropy)"
        "\n\n"
        r"**Dual update:** $\theta^{t+1} = \theta^t + \eta \ell^t$"
        "\n\n"
        r"**Primal recovery:** $x_i = \nabla \Phi^*(\theta) = \frac{e^{\theta_i}}{\sum_j e^{\theta_j}}$ (softmax)"
        "\n\n"
        "**OMD vs FTRL:** Both achieve O(√T) regret. OMD takes a local gradient step "
        "then projects back; FTRL solves a global optimization each round. "
        "For linear losses on the simplex, they coincide when using compatible regularizers."
    )
    params = {
        "eta": {"default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01,
                "help": "Learning rate η (step size in dual space)."},
    }

    def reset(self, n_actions: int, eta: float = 0.1, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.eta = eta
        self.theta = np.zeros(n_actions)  # dual variable

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        return _softmax(self.theta)

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self.theta += self.eta * payoff_vec


class OMD_Euclidean(Strategy):
    name = "OMD (Euclidean)"
    description = (
        "Online Mirror Descent with Euclidean mirror map. "
        "Projected gradient ascent on the simplex."
    )
    theory = (
        r"**Mirror map:** $\Phi(x) = \frac{1}{2}\|x\|_2^2$ (Euclidean)"
        "\n\n"
        r"**Dual update:** $\theta^{t+1} = \theta^t + \eta \ell^t$"
        "\n\n"
        r"**Primal recovery:** $x = \Pi_\Delta(\theta)$ (project onto simplex)"
        "\n\n"
        "Euclidean OMD can concentrate mass sharply on a few actions (sparse solutions), "
        "unlike entropic OMD which keeps all probabilities strictly positive."
    )
    params = {
        "eta": {"default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01,
                "help": "Learning rate η."},
    }

    def reset(self, n_actions: int, eta: float = 0.1, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.eta = eta
        self.theta = np.zeros(n_actions)

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        return _project_simplex(self.theta)

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self.theta += self.eta * payoff_vec


# ---------------------------------------------------------------------------
# Fictitious Play
# ---------------------------------------------------------------------------

class FictitiousPlay(Strategy):
    name = "Fictitious Play"
    description = (
        "Maintains a count of opponent actions and always best-responds to the "
        "opponent's empirical frequency distribution."
    )
    theory = (
        "**Update:** track opponent action counts, then play "
        "argmax_i Σ_j A[i,j] · freq(opp_j)."
        "\n\n"
        "**Convergence:** In zero-sum games, the **time-average** of Fictitious Play "
        "strategies converges to Nash equilibrium (Brown 1951). "
        "However the **instantaneous** strategy may cycle forever."
        "\n\n"
        "**Key difference from Hedge/FTRL:** FP is not a regret minimizer in general. "
        "In general-sum games it can cycle and fail to converge."
        "\n\n"
        "Uses ε-greedy tie-breaking for stability (ε=0.01)."
    )
    params = {
        "eps": {"default": 0.01, "min": 0.0, "max": 0.2, "step": 0.005,
                "help": "Epsilon for soft best-response (blends best-response with uniform)."},
    }

    def reset(self, n_actions: int, eps: float = 0.01, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.eps = eps
        self.opp_counts = np.ones(n_actions)  # uniform prior

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        opp_freq = self.opp_counts / self.opp_counts.sum()
        # Expected payoff for each of my actions against opp's empirical distribution
        ev = payoff_matrix @ opp_freq
        # Soft best response
        best = ev.max()
        br = (ev == best).astype(float)
        br /= br.sum()
        uniform = np.ones(self.n_actions) / self.n_actions
        return (1 - self.eps) * br + self.eps * uniform

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self.opp_counts[opp_action] += 1


# ---------------------------------------------------------------------------
# Best Response to Last Action
# ---------------------------------------------------------------------------

class BestResponse(Strategy):
    name = "Best Response (to last)"
    description = (
        "Plays the pure best response to the opponent's previous action. "
        "Starts with a random action."
    )
    theory = (
        "**Greedy strategy:** observe opponent's last action, play argmax_i A[i, opp_action]."
        "\n\n"
        "**Problem:** Can cycle in rock-paper-scissors. "
        "Against an adaptive opponent, this strategy is predictable and exploitable."
        "\n\n"
        "Contrast with Fictitious Play which best-responds to the **empirical distribution**, "
        "not just the last action — FP is smoother and has convergence guarantees."
    )
    params = {}

    def reset(self, n_actions: int, **kwargs: Any) -> None:
        super().reset(n_actions)
        self._last_opp_action: int | None = None

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        if self._last_opp_action is None:
            return np.ones(self.n_actions) / self.n_actions
        ev = payoff_matrix[:, self._last_opp_action]
        best = ev.max()
        br = (ev == best).astype(float)
        return br / br.sum()

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self._last_opp_action = opp_action


# ---------------------------------------------------------------------------
# Custom (user-defined)
# ---------------------------------------------------------------------------

CUSTOM_TEMPLATE = '''\
import numpy as np

class CustomStrategy:
    """
    Write your own strategy!

    Interface:
      reset(n_actions)              — called once before the game starts
      get_mixed_strategy(payoff_matrix) -> np.ndarray  — return probability vector
      update(my_action, opp_action, payoff_vec)         — update after each round

    payoff_matrix[i, j] = your payoff when you play i and opponent plays j
    payoff_vec[i]        = payoff you'd have got for action i this round
    """

    def reset(self, n_actions: int):
        self.n_actions = n_actions
        self.t = 0
        # --- your state here ---

    def get_mixed_strategy(self, payoff_matrix):
        # Return a probability vector of length n_actions.
        return np.ones(self.n_actions) / self.n_actions

    def update(self, my_action, opp_action, payoff_vec):
        self.t += 1
        # --- update your state here ---
'''


class CustomStrategy(Strategy):
    name = "Custom (code)"
    description = "Write your own strategy in Python."
    theory = (
        "Implement `get_mixed_strategy` and `update` in the code editor. "
        "You can implement any algorithm you like — try implementing Hedge yourself!"
    )
    params = {}

    def reset(self, n_actions: int, code: str = CUSTOM_TEMPLATE, **kwargs: Any) -> None:
        super().reset(n_actions)
        self._strategy = self._compile(code, n_actions)

    @staticmethod
    def _compile(code: str, n_actions: int):
        ns: dict = {}
        exec(compile(code, "<custom_strategy>", "exec"), ns)
        cls = ns.get("CustomStrategy")
        if cls is None:
            raise ValueError("Code must define a class named 'CustomStrategy'.")
        obj = cls()
        obj.reset(n_actions)
        return obj

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        result = self._strategy.get_mixed_strategy(payoff_matrix)
        result = np.asarray(result, dtype=float)
        result = np.clip(result, 0, None)
        total = result.sum()
        if total < 1e-12:
            return np.ones(self.n_actions) / self.n_actions
        return result / total

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self._strategy.update(my_action, opp_action, payoff_vec)


# ---------------------------------------------------------------------------
# Adversarial strategies
# ---------------------------------------------------------------------------

class AdversarialBR(Strategy):
    """
    Stackelberg adversary: observes opponent's current mixed strategy before choosing.
    Requires adversarial_mode=True in the simulation runner.
    """
    name = "Adversarial BR (sees your strategy)"
    description = (
        "Observes your mixed strategy this round *before* choosing, then plays the "
        "pure best response to it. The theoretical worst-case adaptive adversary."
    )
    theory = (
        "**Stackelberg adversary:** In normal simultaneous play, neither player sees the "
        "other's strategy. Here P2 acts as a *leader* who observes P1's mixed strategy "
        "x^t before choosing — equivalent to playing after P1 commits.\n\n"
        "**Why this matters:** Regret bounds for Hedge and FTRL are proven against exactly "
        "this adversary. If R_T / T → 0 here, the algorithm is truly no-regret.\n\n"
        "**Requires:** 'Adversarial mode' toggle in simulation settings — otherwise "
        "this strategy has no information advantage and falls back to uniform."
    )
    params = {
        "eps": {"default": 0.0, "min": 0.0, "max": 0.3, "step": 0.01,
                "help": "Smoothing: blend best-response with uniform (0 = pure BR)."},
    }

    def reset(self, n_actions: int, eps: float = 0.0, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.eps = eps

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        if self._opp_mixed is None:
            # No information yet — play uniform
            return np.ones(self.n_actions) / self.n_actions
        # Best response to opponent's revealed mixed strategy
        ev = payoff_matrix @ self._opp_mixed
        best = ev.max()
        br = (ev == best).astype(float)
        br /= br.sum()
        if self.eps > 0:
            uniform = np.ones(self.n_actions) / self.n_actions
            return (1 - self.eps) * br + self.eps * uniform
        return br


class NashPlayer(Strategy):
    """
    Plays the Nash equilibrium strategy (computed once from the payoff matrix).
    Only valid for zero-sum games.
    """
    name = "Nash Player (minimax)"
    description = (
        "Computes and plays the Nash equilibrium mixed strategy. "
        "In zero-sum games this is the saddle-point / minimax strategy."
    )
    theory = (
        "**Minimax theorem (von Neumann 1928):** In a finite zero-sum game, "
        "max_x min_y x^T A y = min_y max_x x^T A y = V*.\n\n"
        "The Nash strategy guarantees expected payoff ≥ V* regardless of the opponent. "
        "It is *unexploitable* — no adaptive adversary can do better than V* against it.\n\n"
        "**Against a no-regret learner:** The learner's time-average converges toward Nash, "
        "so this match shows what 'convergence' looks like from the stable side.\n\n"
        "Nash is computed via LP (scipy). Falls back to uniform for general-sum games."
    )
    params = {}

    def reset(self, n_actions: int, **kwargs: Any) -> None:
        super().reset(n_actions)
        self._nash: np.ndarray | None = None

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        if self._nash is None:
            self._nash = self._compute_nash(payoff_matrix)
        return self._nash

    @staticmethod
    def _compute_nash(payoff_matrix: np.ndarray) -> np.ndarray:
        """
        Nash for the row player of payoff_matrix (my payoff matrix, minimized by opponent).
        Uses the column-player LP: min_y max_i (B y)_i, where B = payoff_matrix here
        since from *my* perspective payoff_matrix rows = my actions, cols = opp actions.
        We want my Nash mixed strategy.
        """
        from scipy.optimize import linprog
        # payoff_matrix[i,j] = my payoff when I play i, opp plays j.
        # My Nash: max_x min_j (x^T payoff_matrix)_j
        # LP: max v s.t. payoff_matrix^T x >= v*1, sum(x)=1, x>=0
        n, m = payoff_matrix.shape
        c = np.zeros(n + 1); c[-1] = -1.0
        A_ub = np.hstack([-payoff_matrix.T, np.ones((m, 1))])
        b_ub = np.zeros(m)
        A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])
        bounds = [(0.0, None)] * n + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.success:
            return res.x[:n]
        return np.ones(n) / n


class Exploiter(Strategy):
    """
    Tracks opponent's time-average strategy and best-responds to it.
    """
    name = "Exploiter (time-average BR)"
    description = (
        "Tracks the opponent's empirical action frequencies over all past rounds, "
        "then plays the best response to that time-average. "
        "Exploits any systematic bias or pattern in the opponent's strategy."
    )
    theory = (
        "**Time-average exploitation:** If the opponent's strategies x^1, …, x^T "
        "converge (as they should for no-regret learners in zero-sum games), "
        "this adversary quickly identifies and best-responds to the limit.\n\n"
        "**vs. Best Response to last:** BR-to-last reacts to noise; "
        "time-average BR is smoother and more robust.\n\n"
        "**vs. Fictitious Play:** FP is symmetric — both players do this. "
        "Here only P2 is the exploiter, which is a purely adversarial setup.\n\n"
        "**What to expect:** Against Hedge or FTRL converging to Nash, the exploiter's "
        "best response is also Nash, so both players end up near Nash. Against a "
        "biased or cycling opponent the exploiter should win clearly."
    )
    params = {
        "warmup": {"default": 10, "min": 1, "max": 100, "step": 1,
                   "help": "Rounds of uniform play before switching to exploitation."},
    }

    def reset(self, n_actions: int, warmup: int = 10, **kwargs: Any) -> None:
        super().reset(n_actions)
        self.warmup = warmup
        self._opp_action_counts = np.ones(n_actions)  # Laplace smoothing
        self._t = 0

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        if self._t < self.warmup:
            return np.ones(self.n_actions) / self.n_actions
        opp_freq = self._opp_action_counts / self._opp_action_counts.sum()
        ev = payoff_matrix @ opp_freq
        best = ev.max()
        br = (ev == best).astype(float)
        return br / br.sum()

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self._opp_action_counts[opp_action] += 1
        self._t += 1


class Cycler(Strategy):
    """
    Cycles through actions in a fixed order. Predictable and exploitable.
    """
    name = "Cycler (fixed pattern)"
    description = (
        "Plays actions 0, 1, 2, … in a fixed repeating cycle. "
        "Highly predictable — any learner should eventually exploit this."
    )
    theory = (
        "**Predictable opponent:** Cycler is deterministic and periodic. "
        "Against it, Fictitious Play and Exploiter should converge to a pure BR. "
        "Hedge converges more slowly (it doesn't *predict* the cycle, just tracks it).\n\n"
        "**Regret perspective:** The cycler itself has high regret — it ignores all feedback. "
        "Use it to see how quickly each learning algorithm detects and exploits a pattern.\n\n"
        "**Interesting case:** In RPS, cycling R→P→S→R is actually a Nash strategy "
        "in time-average (1/3 each action), so Hedge won't gain much against it!"
    )
    params = {
        "offset": {"default": 0, "min": 0, "max": 10, "step": 1,
                   "help": "Starting action index in the cycle."},
    }

    def reset(self, n_actions: int, offset: int = 0, **kwargs: Any) -> None:
        super().reset(n_actions)
        self._step = offset % n_actions

    def get_mixed_strategy(self, payoff_matrix: np.ndarray) -> np.ndarray:
        x = np.zeros(self.n_actions)
        x[self._step] = 1.0
        return x

    def update(self, my_action: int, opp_action: int, payoff_vec: np.ndarray) -> None:
        self._step = (self._step + 1) % self.n_actions


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_STRATEGIES: dict[str, type[Strategy]] = {
    "Uniform Random": Uniform,
    "Fixed Pure": PureStrategy,
    "Hedge (MWU)": Hedge,
    "FTRL (Entropy)": FTRL_Entropy,
    "FTRL (L2 / Euclidean)": FTRL_L2,
    "OMD (Entropic)": OMD_Entropic,
    "OMD (Euclidean)": OMD_Euclidean,
    "Fictitious Play": FictitiousPlay,
    "Best Response (to last)": BestResponse,
    "Custom (code)": CustomStrategy,
    # --- adversarial ---
    "Adversarial BR (sees your strategy)": AdversarialBR,
    "Nash Player (minimax)": NashPlayer,
    "Exploiter (time-average BR)": Exploiter,
    "Cycler (fixed pattern)": Cycler,
}
