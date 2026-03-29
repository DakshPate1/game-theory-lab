# Game Theory Lab

An interactive sandbox for two-player games. Pick a game, assign an algorithm to each player, run the simulation, and watch mixed strategies evolve, payoffs accumulate, and regret converge — or not.

Built for learning the online learning algorithms covered in RL theory: **Hedge**, **FTRL**, **OMD**, **Fictitious Play**, and adversarial variants. Every algorithm's update rule and regret bound is shown directly in the UI.

```
streamlit run app.py
```

---

## Concepts, in order

### 1 · What is a game?

A [normal-form game](https://en.wikipedia.org/wiki/Normal-form_game) between two players is defined by:
- A finite set of **actions** for each player
- A **payoff function** mapping action pairs to outcomes

Here we represent payoffs as a matrix **A** (rows = Player 1 actions, columns = Player 2 actions), where `A[i,j]` is P1's payoff when P1 plays action `i` and P2 plays action `j`.

**In this project:** `games.py` defines five preset games as payoff matrices: Rock Paper Scissors, Matching Pennies, Biased RPS, Prisoner's Dilemma, and Battle of the Sexes, plus a custom editor.

---

### 2 · Zero-sum games

A game is [zero-sum](https://en.wikipedia.org/wiki/Zero-sum_game) if one player's gain is exactly the other's loss: `A[i,j] + B[j,i] = 0` for all `(i,j)`, where **B** is P2's payoff matrix. Equivalently, `B = -Aᵀ`.

Zero-sum games model pure competition: poker, RPS, matching pennies. They have especially clean theory because the interests of the two players are perfectly opposed.

**Non-zero-sum games** (Prisoner's Dilemma, Battle of the Sexes) allow mutual gain or mutual loss. The theory is harder and regret minimizers may not converge to Nash in these games.

---

### 3 · Mixed strategies and the Minimax Theorem

A [mixed strategy](https://en.wikipedia.org/wiki/Strategy_(game_theory)#Mixed_strategy) is a probability distribution over actions. Player 1 plays action `i` with probability `x[i]`, where `x ∈ Δ` (the probability simplex).

Why mix? Because pure strategies are predictable and exploitable. In RPS, always playing Rock loses to Paper every time. The solution is to randomize.

[von Neumann's Minimax Theorem (1928)](https://en.wikipedia.org/wiki/Minimax_theorem) states that in any finite two-player zero-sum game:

$$\max_{x \in \Delta} \min_{y \in \Delta} \; x^\top A y \;\;=\;\; \min_{y \in \Delta} \max_{x \in \Delta} \; x^\top A y \;\;=\;\; V^*$$

The shared value `V*` is the **game value**: the expected payoff P1 can guarantee regardless of P2's strategy, and the maximum P2 can hold P1 to.

**In this project:** The Nash gap / exploitability plot shows how far each player's current strategy is from achieving `V*`.

---

### 4 · Nash Equilibrium

A [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) is a pair of strategies `(x*, y*)` where neither player can improve by deviating unilaterally:

$$x^* \in \arg\max_x \; x^\top A y^*, \qquad y^* \in \arg\min_y \; (x^*)^\top A y$$

For zero-sum games, Nash equilibria always exist (minimax theorem), are unique in game value, and can be computed by [linear programming](https://en.wikipedia.org/wiki/Linear_programming).

**In this project:** `games.py` computes Nash for zero-sum games via `scipy.optimize.linprog`. The Nash strategy for each player is shown in the expandable panel, and the exploitability chart measures how far the players' time-average strategies are from Nash.

---

### 5 · Online learning and regret

The core question: can an algorithm learn to play well **without knowing the opponent's strategy in advance?**

[Online learning](https://en.wikipedia.org/wiki/Online_machine_learning) frames this as a sequential decision problem. At each round `t`:
1. The algorithm chooses a mixed strategy `xᵗ`
2. The environment reveals a loss vector `ℓᵗ` (or equivalently, the opponent's action)
3. The algorithm observes `ℓᵗ` and updates

The algorithm's **external regret** after `T` rounds is how much it lost compared to the best single fixed action in hindsight:

$$R_T = \max_{i \in \text{actions}} \sum_{t=1}^T \ell_i^t \;-\; \sum_{t=1}^T \ell_{a_t}^t$$

An algorithm is a **no-regret learner** if `R_T / T → 0` as `T → ∞`.

**Why does this matter?** If both players run no-regret algorithms in a zero-sum game, their time-average strategies converge to Nash equilibrium. This gives an algorithmic path to Nash without solving a global optimization.

**In this project:** The regret plot tracks `R_T / T` per round for each player, overlaid with the theoretical `O(√(ln n / T))` bound for Hedge.

---

### 6 · Hedge / Multiplicative Weights Update

[Hedge](https://en.wikipedia.org/wiki/Multiplicative_weights_update_method) (also called MWU or EXP3 in the bandit setting) is the canonical no-regret algorithm for the simplex.

**Update rule:** maintain weights `wᵢ` for each action. After observing payoff vector `ℓᵗ`:

$$w_i^{t+1} = w_i^t \cdot e^{\,\eta\, \ell_i^t}, \qquad x^{t+1} = \frac{w^{t+1}}{\|w^{t+1}\|_1}$$

Actions that paid off get higher weight; the distribution tilts toward them. `η` (learning rate) controls how aggressively weights shift.

**Regret bound:**

$$R_T \;\leq\; \frac{\ln n}{\eta} + \eta T \qquad \Longrightarrow \qquad \text{set } \eta = \sqrt{\tfrac{\ln n}{T}}: \quad R_T = O\!\left(\sqrt{T \ln n}\right)$$

The bound scales as `O(1/√T)` per round — sublinear, so it goes to 0.

**In this project:** `Hedge` in `algorithms.py`, implemented in log-space for numerical stability. You can tune `η` with a slider and watch how larger values converge faster but oscillate more.

---

### 7 · FTRL — Follow the Regularized Leader

[FTRL](https://en.wikipedia.org/wiki/Follow_the_regularized_leader) is the general framework that contains Hedge as a special case.

At each round, FTRL picks the strategy that minimizes the cumulative loss so far plus a regularization term:

$$x^{t+1} = \arg\min_{x \in \Delta} \left[ \eta \left\langle \sum_{s \leq t} \ell^s,\; x \right\rangle + R(x) \right]$$

The regularizer `R(x)` prevents the algorithm from swinging too aggressively to the best past action.

| Regularizer | Closed-form solution | Algorithm |
|---|---|---|
| `R(x) = Σ xᵢ ln xᵢ` (negative entropy) | softmax of cumulative payoffs | **= Hedge** |
| `R(x) = ½‖x‖₂²` (squared L2 norm) | project cumulative payoffs onto simplex | projected gradient |

**In this project:** `FTRL_Entropy` and `FTRL_L2` in `algorithms.py`. Running both side-by-side on the same game shows how the regularizer shapes convergence: entropy keeps all probabilities positive (full support), while L2 can drive some to exactly zero.

---

### 8 · OMD — Online Mirror Descent

[Online Mirror Descent](https://en.wikipedia.org/wiki/Mirror_descent) is an alternative derivation of the same family of algorithms. Instead of minimizing a global objective (FTRL), OMD takes a local gradient step in a *dual space* and maps back to the primal.

**Algorithm:**
1. Dual update: `θᵗ⁺¹ = θᵗ + η ℓᵗ` (gradient step in dual space)
2. Primal recovery: `xᵗ⁺¹ = ∇Φ*(θᵗ⁺¹)` via the conjugate of mirror map `Φ`

| Mirror map `Φ` | Conjugate `∇Φ*` | Algorithm |
|---|---|---|
| `Σ xᵢ ln xᵢ` (entropic) | softmax(θ) | **= Hedge** |
| `½‖x‖₂²` (Euclidean) | Π_Δ(θ) (project onto simplex) | projected gradient |

**Key insight:** FTRL and OMD are *different algorithms* — one minimizes globally, the other takes local steps — yet they produce identical iterates when using compatible regularizers. The UI shows this: `Hedge`, `FTRL (Entropy)`, and `OMD (Entropic)` produce the same convergence curve.

**In this project:** `OMD_Entropic` and `OMD_Euclidean` in `algorithms.py`.

---

### 9 · Fictitious Play

[Fictitious Play](https://en.wikipedia.org/wiki/Fictitious_play) is an older algorithm from 1951 with a different motivation: each player maintains the empirical frequency of the opponent's past actions and best-responds to it.

**Update:** observe opponent's action → increment count → play `argmax_i Σⱼ A[i,j] · freq(opp_j)`

**Convergence:** In zero-sum games, the time-average of both players' strategies converges to Nash (Brown 1951). But the *instantaneous* strategy cycles and the algorithm is **not** a regret minimizer in general.

**Why it's in here:** FP has a different flavor from Hedge/FTRL — it models the opponent explicitly rather than minimizing regret. Running `Fictitious Play vs Cycler` on RPS shows a known failure mode: FP gets stuck predicting the wrong next move and loses about 1/3 of the game value per round.

**In this project:** `FictitiousPlay` in `algorithms.py`.

---

### 10 · Adversarial opponents

[Adversarial online learning](https://en.wikipedia.org/wiki/Online_machine_learning#Adversarial_online_learning) asks: what if the environment is trying to maximize your regret?

The standard regret bounds for Hedge/FTRL/OMD hold against an *adaptive adversary* — one who sees your past actions and adapts. An even stronger adversary is the **Stackelberg adversary** who sees your *current mixed strategy* `xᵗ` before choosing their action. This is the theoretical worst case.

The lab includes four adversarial strategies:

| Strategy | What it does | What you learn |
|---|---|---|
| **Adversarial BR** | Observes your mixed strategy before choosing (Stackelberg) | Hedge's regret bound holds even here |
| **Nash Player** | Always plays Nash equilibrium strategy | No learner can exploit Nash; it's a fixed point |
| **Exploiter** | Best-responds to your time-average distribution | Detects and exploits systematic bias |
| **Cycler** | Plays actions in a fixed loop | Pattern-exploitation; breaks Fictitious Play |

**Adversarial mode** in the simulation settings activates the Stackelberg setup: P2 observes P1's mixed strategy before choosing. The key experiment: run `Hedge vs Adversarial BR` in adversarial mode — Hedge's regret should still match `O(√(ln n / T))`.

---

### 11 · Why pit strategies against each other?

The Nash convergence theorem says: **if both players run no-regret algorithms in a zero-sum game, their time-average strategies converge to Nash equilibrium.**

Formal statement: if P1 has regret `R₁(T)` and P2 has regret `R₂(T)`, and `(x̄, ȳ)` are their time-average strategies, then:

$$\max_x \; x^\top A \bar{y} \;-\; \min_y \; \bar{x}^\top A y \;\;\leq\;\; \frac{R_1(T) + R_2(T)}{T}$$

The left side is the Nash gap (how far `(x̄, ȳ)` is from equilibrium). As regret grows sublinearly, the gap shrinks to 0.

This is why the simulation pits algorithms against each other — we need *two* players to observe equilibrium-finding in action. A single no-regret algorithm adapting to a fixed opponent won't converge to Nash; it just learns to exploit the opponent. Put two adaptive learners together and Nash emerges from their interaction.

---

## Project structure

```
game_theory_lab/
├── app.py          Streamlit UI — 3 tabs: Sandbox, Algorithm Explorer, Theory Notes
├── games.py        Game definitions; Nash LP solver; exploitability computation
├── algorithms.py   All strategy implementations + custom code editor
├── simulation.py   Simulation runner; regret tracking; exploitability
└── requirements.txt
```

**`games.py`**
- `GAMES` dict: payoff matrices for all preset games
- `compute_nash_zero_sum(A)`: LP via `scipy.optimize.linprog` → `(nash_row, nash_col, value)`
- `exploitability(x, y, A, B)`: Nash gap for time-average strategies

**`algorithms.py`**
- `Strategy` base class: `reset / get_mixed_strategy / update / observe_opponent_strategy`
- Learning algorithms: `Hedge`, `FTRL_Entropy`, `FTRL_L2`, `OMD_Entropic`, `OMD_Euclidean`
- Classical: `FictitiousPlay`, `BestResponse`, `Uniform`, `PureStrategy`
- Adversarial: `AdversarialBR`, `NashPlayer`, `Exploiter`, `Cycler`
- `CustomStrategy`: exec-based code editor with full interface access

**`simulation.py`**
- Simultaneous play loop: both players choose before either updates
- Regret tracking via cumulative counterfactual payoffs
- `adversarial_p2` / `adversarial_p1` flags for Stackelberg mode

---

## Suggested experiments

| Experiment | Setup | What to look for |
|---|---|---|
| Nash convergence | Hedge vs Hedge, RPS, η=0.05 | Time-average → (1/3, 1/3, 1/3); exploitability → 0 |
| Learning rate sensitivity | Hedge vs Hedge, vary η | Large η: fast but noisy; small η: slow but smooth |
| FTRL = Hedge | FTRL (Entropy) vs Hedge, any game | Identical regret curves — same algorithm, two derivations |
| L2 vs Entropy | FTRL (L2) vs FTRL (Entropy) | L2 concentrates on fewer actions; can hit zero probability |
| Adversarial robustness | Hedge vs Adversarial BR + adversarial mode | Regret still ≈ O(√(ln n / T)) — the bound holds |
| Unexploitability of Nash | Uniform vs Adversarial BR, Matching Pennies | Nash = uniform → adversary can't gain edge |
| FP failure mode | FP vs Cycler, RPS | FP loses ~1/3 per round — not a regret minimizer |
| General-sum breakdown | Hedge vs Hedge, Prisoner's Dilemma | Convergence to Nash not guaranteed; observe what happens |

---

## Dependencies

```
streamlit   plotly   numpy   scipy   pandas
```

Install: `pip install -r requirements.txt`

Python ≥ 3.11 recommended (`pyenv local 3.11.9` if using pyenv).

---

## Further reading

**Foundations**
- [Normal-form game](https://en.wikipedia.org/wiki/Normal-form_game)
- [Zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game)
- [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium)
- [Minimax theorem](https://en.wikipedia.org/wiki/Minimax_theorem)
- [Mixed strategy](https://en.wikipedia.org/wiki/Strategy_(game_theory)#Mixed_strategy)

**Online learning**
- [Regret (online learning)](https://en.wikipedia.org/wiki/Regret_(decision_theory))
- [Online machine learning](https://en.wikipedia.org/wiki/Online_machine_learning)
- [Multiplicative weights update](https://en.wikipedia.org/wiki/Multiplicative_weights_update_method) (Hedge / MWU)
- [Follow the regularized leader](https://en.wikipedia.org/wiki/Follow_the_regularized_leader)
- [Mirror descent](https://en.wikipedia.org/wiki/Mirror_descent)

**Classical game theory**
- [Fictitious play](https://en.wikipedia.org/wiki/Fictitious_play)
- [Best response](https://en.wikipedia.org/wiki/Best_response)
- [Correlated equilibrium](https://en.wikipedia.org/wiki/Correlated_equilibrium) (next step: swap regret minimization)

**Specific games**
- [Rock paper scissors](https://en.wikipedia.org/wiki/Rock_paper_scissors)
- [Matching pennies](https://en.wikipedia.org/wiki/Matching_pennies)
- [Prisoner's dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma)
- [Battle of the sexes](https://en.wikipedia.org/wiki/Battle_of_the_sexes_(game_theory))
