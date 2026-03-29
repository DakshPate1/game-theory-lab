"""
Game Theory Lab — interactive sandbox for two-player games.

Run with:  streamlit run app.py
"""

import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from games import GAMES, get_payoff_B, compute_nash_zero_sum, exploitability
from algorithms import ALL_STRATEGIES, CUSTOM_TEMPLATE, CustomStrategy
from simulation import run_simulation, SimulationResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Game Theory Lab",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("♟ Game Theory Lab")
st.caption(
    "Interactive sandbox for two-player games. "
    "Select a game, choose algorithms for each player, run the simulation."
)

# ---------------------------------------------------------------------------
# Tabs: Sandbox | Algorithm Explorer
# ---------------------------------------------------------------------------
tab_sandbox, tab_algo, tab_theory = st.tabs(
    ["🎮 Sandbox", "📖 Algorithm Explorer", "📐 Theory Notes"]
)

# ============================================================
# SANDBOX TAB
# ============================================================
with tab_sandbox:

    # ---- Game selection ----
    st.subheader("1 · Game")
    col_game, col_desc = st.columns([1, 2])

    with col_game:
        game_name = st.selectbox("Select game", list(GAMES.keys()), key="game_name")

    game = dict(GAMES[game_name])  # copy so we can mutate for Custom

    with col_desc:
        st.info(game["description"])
        tag = "✅ Zero-sum" if game["zero_sum"] else "⚠️ General-sum (not zero-sum)"
        st.caption(tag)

    # ---- Custom payoff matrix editor ----
    if game_name == "Custom":
        with st.expander("Edit payoff matrix (P1's payoffs)"):
            st.caption(
                "Row = P1's action, Column = P2's action. "
                "Enter P1's payoff for each cell. "
                "Check 'zero-sum' to auto-fill P2's matrix as −A."
            )
            n_actions = st.slider("Number of actions per player", 2, 6, 2, key="custom_n")

            # Action names
            col_a, col_b = st.columns(2)
            with col_a:
                p1_names = [
                    st.text_input(f"P1 action {i}", value=f"A{i}", key=f"p1n{i}")
                    for i in range(n_actions)
                ]
            with col_b:
                p2_names = [
                    st.text_input(f"P2 action {j}", value=f"B{j}", key=f"p2n{j}")
                    for j in range(n_actions)
                ]

            # Editable dataframe for payoff matrix
            default_A = pd.DataFrame(
                np.eye(n_actions) * 2 - 1,
                index=p1_names,
                columns=p2_names,
            )
            edited = st.data_editor(default_A, key="custom_matrix")
            payoff_A = edited.values.astype(float)

            is_zs = st.checkbox("Zero-sum game", value=True, key="custom_zs")
            game["payoff_A"] = payoff_A
            game["zero_sum"] = is_zs
            game["actions_p1"] = p1_names
            game["actions_p2"] = p2_names
            if not is_zs:
                st.caption("P2's payoff matrix (rows = P2 actions, cols = P1 actions):")
                default_B = pd.DataFrame(
                    -payoff_A.T,
                    index=p2_names,
                    columns=p1_names,
                )
                edited_B = st.data_editor(default_B, key="custom_matrix_B")
                game["payoff_B"] = edited_B.values.astype(float)

    # Show payoff matrix
    with st.expander("Show payoff matrix"):
        payoff_A = game["payoff_A"]
        payoff_B = get_payoff_B(game)
        actions_p1 = game["actions_p1"]
        actions_p2 = game["actions_p2"]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**P1's payoffs** (row = P1, col = P2)")
            st.dataframe(
                pd.DataFrame(payoff_A, index=actions_p1, columns=actions_p2),
                use_container_width=True,
            )
        with col_m2:
            st.markdown("**P2's payoffs** (row = P2, col = P1)")
            st.dataframe(
                pd.DataFrame(payoff_B, index=actions_p2, columns=actions_p1),
                use_container_width=True,
            )

    # Nash equilibrium (zero-sum only)
    if game["zero_sum"]:
        try:
            nash_row, nash_col, game_value = compute_nash_zero_sum(game["payoff_A"])
            with st.expander("Nash equilibrium (zero-sum LP solution)"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**P1's Nash strategy**")
                    for a, p in zip(actions_p1, nash_row):
                        st.write(f"{a}: {p:.3f}")
                with c2:
                    st.markdown("**P2's Nash strategy**")
                    for a, p in zip(actions_p2, nash_col):
                        st.write(f"{a}: {p:.3f}")
                with c3:
                    st.metric("Game value", f"{game_value:.4f}")
                    st.caption("(P1's expected payoff at Nash)")
        except Exception as e:
            st.warning(f"Nash computation failed: {e}")

    st.divider()

    # ---- Player configuration ----
    st.subheader("2 · Players")

    def player_config(player_num: int, default_algo: str, default_key: str):
        st.markdown(f"**Player {player_num}**")
        algo_name = st.selectbox(
            "Algorithm", list(ALL_STRATEGIES.keys()),
            index=list(ALL_STRATEGIES.keys()).index(default_algo),
            key=f"algo_{player_num}",
        )
        cls = ALL_STRATEGIES[algo_name]
        instance = cls()

        # Theory tooltip
        if instance.theory:
            with st.expander("Theory"):
                st.markdown(instance.theory)

        # Hyperparameter sliders
        kwargs: dict = {}
        for param, meta in instance.params.items():
            if param == "action":
                n = game["payoff_A"].shape[player_num - 1]
                kwargs[param] = st.slider(
                    meta.get("help", param),
                    0, n - 1, min(meta["default"], n - 1),
                    key=f"p{player_num}_{param}",
                )
            else:
                kwargs[param] = st.slider(
                    meta.get("help", param),
                    float(meta["min"]), float(meta["max"]),
                    float(meta["default"]), float(meta["step"]),
                    key=f"p{player_num}_{param}",
                )

        # Custom code editor
        code = CUSTOM_TEMPLATE
        if algo_name == "Custom (code)":
            code = st.text_area(
                "Strategy code",
                value=CUSTOM_TEMPLATE,
                height=320,
                key=f"code_{player_num}",
            )
            kwargs["code"] = code

        return algo_name, cls, kwargs

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1_algo_name, p1_cls, p1_kwargs = player_config(1, "Hedge (MWU)", "p1")
    with col_p2:
        p2_algo_name, p2_cls, p2_kwargs = player_config(2, "Uniform Random", "p2")

    st.divider()

    # ---- Simulation settings ----
    st.subheader("3 · Simulation")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        n_rounds = st.slider("Rounds", 100, 10_000, 2_000, 100, key="n_rounds")
    with col_s2:
        seed = st.number_input("Random seed", value=42, min_value=0, key="seed")
    with col_s3:
        exploit_every = st.slider(
            "Nash-gap check every N rounds", 0, 200, 50, 10, key="exploit_every",
            help="0 to disable (faster). Computes exploitability of time-average strategies.",
        )

    with st.expander("⚔️ Adversarial mode"):
        st.markdown(
            "In standard play, both players choose **simultaneously** — neither sees "
            "the other's strategy this round. Adversarial mode breaks this: one player "
            "acts as a **Stackelberg leader**, observing the opponent's mixed strategy "
            "before choosing. This is the theoretical worst-case setting that "
            "regret bounds (Hedge, FTRL, OMD) are proven against."
        )
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            adversarial_p2 = st.checkbox(
                "P2 sees P1's strategy first",
                value=False, key="adv_p2",
                help="P2 observes P1's mixed strategy this round before choosing. "
                     "Use with 'Adversarial BR' as P2 to test P1's regret bound.",
            )
        with adv_col2:
            adversarial_p1 = st.checkbox(
                "P1 sees P2's strategy first",
                value=False, key="adv_p1",
                help="P1 observes P2's mixed strategy this round before choosing.",
            )
        if adversarial_p2 and adversarial_p1:
            st.warning("Both can't see each other simultaneously — P2-sees-P1 takes precedence.")
            adversarial_p1 = False

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        try:
            n_actions_p1 = game["payoff_A"].shape[0]
            n_actions_p2 = game["payoff_A"].shape[1]

            p1 = p1_cls()
            p1.reset(n_actions_p1, **p1_kwargs)
            p2 = p2_cls()
            p2.reset(n_actions_p2, **p2_kwargs)

            with st.spinner("Running simulation..."):
                result = run_simulation(
                    game, p1, p2,
                    n_rounds=n_rounds,
                    seed=int(seed),
                    exploit_every=int(exploit_every),
                    adversarial_p2=adversarial_p2,
                    adversarial_p1=adversarial_p1,
                )
            st.session_state["result"] = result
            st.session_state["game"] = game
            st.session_state["p1_algo"] = p1_algo_name
            st.session_state["p2_algo"] = p2_algo_name
            st.success(f"Done — {n_rounds} rounds simulated.")
        except Exception:
            st.error("Simulation failed:")
            st.code(traceback.format_exc())

    # ---- Results ----
    if "result" in st.session_state:
        result: SimulationResult = st.session_state["result"]
        game_r = st.session_state["game"]
        p1_name = st.session_state["p1_algo"]
        p2_name = st.session_state["p2_algo"]
        actions_p1 = game_r["actions_p1"]
        actions_p2 = game_r["actions_p2"]
        T = len(result.p1_payoffs)
        rounds = np.arange(1, T + 1)

        st.divider()
        st.subheader("4 · Results")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("P1 total payoff", f"{sum(result.p1_payoffs):.1f}")
        with c2:
            st.metric("P2 total payoff", f"{sum(result.p2_payoffs):.1f}")
        with c3:
            final_regret_p1 = result.p1_regret[-1] if len(result.p1_regret) else 0
            st.metric("P1 avg regret/round", f"{final_regret_p1:.4f}")
        with c4:
            final_regret_p2 = result.p2_regret[-1] if len(result.p2_regret) else 0
            st.metric("P2 avg regret/round", f"{final_regret_p2:.4f}")

        # Plot tabs
        ptab1, ptab2, ptab3, ptab4 = st.tabs([
            "📊 Strategy Evolution",
            "💰 Cumulative Payoff",
            "📉 Regret",
            "🎯 Nash Gap",
        ])

        with ptab1:
            st.markdown("Probability each player assigns to each action over time (mixed strategy).")
            col_ev1, col_ev2 = st.columns(2)
            with col_ev1:
                st.markdown(f"**{p1_name}** (P1)")
                p1_mix = np.array(result.p1_mixed)
                fig = go.Figure()
                colors = px.colors.qualitative.Set2
                for i, a in enumerate(actions_p1):
                    fig.add_trace(go.Scatter(
                        x=rounds, y=p1_mix[:, i],
                        name=a, mode="lines",
                        line=dict(color=colors[i % len(colors)], width=1.5),
                    ))
                fig.update_layout(
                    yaxis=dict(title="Probability", range=[0, 1]),
                    xaxis_title="Round",
                    legend_title="Action",
                    height=320, margin=dict(t=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_ev2:
                st.markdown(f"**{p2_name}** (P2)")
                p2_mix = np.array(result.p2_mixed)
                fig2 = go.Figure()
                for j, a in enumerate(actions_p2):
                    fig2.add_trace(go.Scatter(
                        x=rounds, y=p2_mix[:, j],
                        name=a, mode="lines",
                        line=dict(color=colors[j % len(colors)], width=1.5),
                    ))
                fig2.update_layout(
                    yaxis=dict(title="Probability", range=[0, 1]),
                    xaxis_title="Round",
                    legend_title="Action",
                    height=320, margin=dict(t=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Time-average
            st.markdown("**Time-average strategies** (converge to Nash for zero-sum regret minimizers)")
            avg_p1 = result.p1_avg_mixed
            avg_p2 = result.p2_avg_mixed
            col_ta1, col_ta2 = st.columns(2)
            with col_ta1:
                fig_bar = go.Figure(go.Bar(
                    x=actions_p1, y=avg_p1,
                    marker_color=[colors[i % len(colors)] for i in range(len(actions_p1))],
                ))
                fig_bar.update_layout(
                    title="P1 time-average", yaxis_range=[0, 1], height=250, margin=dict(t=30),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_ta2:
                fig_bar2 = go.Figure(go.Bar(
                    x=actions_p2, y=avg_p2,
                    marker_color=[colors[j % len(colors)] for j in range(len(actions_p2))],
                ))
                fig_bar2.update_layout(
                    title="P2 time-average", yaxis_range=[0, 1], height=250, margin=dict(t=30),
                )
                st.plotly_chart(fig_bar2, use_container_width=True)

        with ptab2:
            st.markdown(
                "Cumulative payoff over time. "
                "In a zero-sum game, P1 + P2 payoffs should sum to near zero."
            )
            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(
                x=rounds, y=result.p1_cum_payoff,
                name=f"P1 ({p1_name})", mode="lines",
                line=dict(color="royalblue", width=2),
            ))
            fig_pay.add_trace(go.Scatter(
                x=rounds, y=result.p2_cum_payoff,
                name=f"P2 ({p2_name})", mode="lines",
                line=dict(color="tomato", width=2),
            ))
            fig_pay.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_pay.update_layout(
                xaxis_title="Round", yaxis_title="Cumulative payoff",
                height=360, margin=dict(t=20),
            )
            st.plotly_chart(fig_pay, use_container_width=True)

            # Per-round payoff (smoothed)
            window = max(1, T // 100)
            p1_smooth = pd.Series(result.p1_payoffs).rolling(window, min_periods=1).mean()
            p2_smooth = pd.Series(result.p2_payoffs).rolling(window, min_periods=1).mean()
            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scatter(
                x=rounds, y=p1_smooth, name=f"P1 ({p1_name})",
                mode="lines", line=dict(color="royalblue", width=1.5),
            ))
            fig_rp.add_trace(go.Scatter(
                x=rounds, y=p2_smooth, name=f"P2 ({p2_name})",
                mode="lines", line=dict(color="tomato", width=1.5),
            ))
            fig_rp.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_rp.update_layout(
                title=f"Per-round payoff (rolling mean, window={window})",
                xaxis_title="Round", yaxis_title="Payoff",
                height=300, margin=dict(t=40),
            )
            st.plotly_chart(fig_rp, use_container_width=True)

        with ptab3:
            st.markdown(
                r"**Average external regret** at round T: "
                r"$R_T/T = \frac{1}{T}\left(\max_i \sum_t \ell_i^t - \sum_t \ell_{a_t}^t\right)$"
                "\n\nSublinear regret (→ 0) means the algorithm is a **no-regret** learner. "
                "For Hedge, theory predicts $R_T/T = O(1/\sqrt{T})$."
            )
            p1_reg = result.p1_regret
            p2_reg = result.p2_regret
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=rounds, y=p1_reg, name=f"P1 ({p1_name})",
                mode="lines", line=dict(color="royalblue", width=2),
            ))
            fig_reg.add_trace(go.Scatter(
                x=rounds, y=p2_reg, name=f"P2 ({p2_name})",
                mode="lines", line=dict(color="tomato", width=2),
            ))
            # theoretical bound for hedge
            if "Hedge" in p1_name or "FTRL" in p1_name or "OMD" in p1_name:
                import math
                n1 = len(actions_p1)
                # bound: sqrt(log(n)/T) (assumes payoffs in [-1,1])
                bound_p1 = np.sqrt(math.log(n1) / rounds)
                fig_reg.add_trace(go.Scatter(
                    x=rounds, y=bound_p1,
                    name=f"O(√(ln {n1}/T)) bound",
                    mode="lines", line=dict(color="royalblue", dash="dash", width=1),
                    opacity=0.5,
                ))
            fig_reg.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_reg.update_layout(
                xaxis_title="Round", yaxis_title="Avg regret / round",
                height=360, margin=dict(t=20),
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        with ptab4:
            if not result.p1_exploit:
                st.info("Set 'Nash-gap check' > 0 in simulation settings to see exploitability.")
            else:
                st.markdown(
                    "**Exploitability** of the time-average strategy up to round T. "
                    "Measures how much each player could gain by deviating unilaterally. "
                    "Should → 0 for regret minimizers in zero-sum games (Nash convergence)."
                )
                n_points = len(result.p1_exploit)
                exploit_rounds = np.linspace(exploit_every, T, n_points, dtype=int)
                fig_ex = go.Figure()
                fig_ex.add_trace(go.Scatter(
                    x=exploit_rounds, y=result.p1_exploit,
                    name=f"P1 ({p1_name})",
                    mode="lines+markers", line=dict(color="royalblue", width=2),
                    marker=dict(size=4),
                ))
                fig_ex.add_trace(go.Scatter(
                    x=exploit_rounds, y=result.p2_exploit,
                    name=f"P2 ({p2_name})",
                    mode="lines+markers", line=dict(color="tomato", width=2),
                    marker=dict(size=4),
                ))
                fig_ex.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_ex.update_layout(
                    xaxis_title="Round", yaxis_title="Exploitability (Nash gap)",
                    height=360, margin=dict(t=20),
                )
                st.plotly_chart(fig_ex, use_container_width=True)

                if game_r["zero_sum"]:
                    try:
                        nash_row, nash_col, game_value = compute_nash_zero_sum(game_r["payoff_A"])
                        avg_p1 = result.p1_avg_mixed
                        avg_p2 = result.p2_avg_mixed
                        payoff_B_r = get_payoff_B(game_r)
                        eps1_final, eps2_final = exploitability(
                            avg_p1, avg_p2, game_r["payoff_A"], payoff_B_r
                        )
                        st.caption(
                            f"Final exploitability — P1: {eps1_final:.5f}, P2: {eps2_final:.5f} "
                            f"| Nash game value: {game_value:.4f}"
                        )
                    except Exception:
                        pass


# ============================================================
# ALGORITHM EXPLORER TAB
# ============================================================
with tab_algo:
    st.subheader("Algorithm Explorer")
    st.markdown(
        "Explore each algorithm's update rule, convergence properties, "
        "and how it behaves on a simple game."
    )

    selected = st.selectbox(
        "Algorithm", list(ALL_STRATEGIES.keys()), key="explorer_algo"
    )
    cls = ALL_STRATEGIES[selected]
    inst = cls()

    col_d, col_t = st.columns([1, 2])
    with col_d:
        st.markdown(f"**{inst.name}**")
        st.write(inst.description)
        if inst.params:
            st.markdown("**Hyperparameters:**")
            for k, v in inst.params.items():
                st.markdown(f"- `{k}`: {v.get('help', '')} (default {v['default']})")

    with col_t:
        if inst.theory:
            st.markdown(inst.theory)

    st.divider()

    # Quick experiment: run selected algo vs Uniform on Matching Pennies
    st.markdown(
        "**Quick experiment:** this algorithm (P1) vs **Uniform Random** (P2) "
        "on **Matching Pennies** for 2000 rounds."
    )

    exp_eta = None
    if "eta" in inst.params:
        exp_eta = st.slider(
            "Learning rate η", 0.001, 2.0, float(inst.params["eta"]["default"]), 0.01,
            key="exp_eta",
        )

    if st.button("Run quick experiment", key="exp_run"):
        from algorithms import Uniform
        from games import GAMES

        mp_game = GAMES["Matching Pennies"]
        p1_exp = cls()
        kwargs_exp = {}
        if exp_eta is not None:
            kwargs_exp["eta"] = exp_eta
        if selected == "Custom (code)":
            kwargs_exp["code"] = CUSTOM_TEMPLATE
        p1_exp.reset(2, **kwargs_exp)

        p2_exp = Uniform()
        p2_exp.reset(2)

        res_exp = run_simulation(mp_game, p1_exp, p2_exp, n_rounds=2000, seed=0, exploit_every=50)
        st.session_state["exp_result"] = res_exp

    if "exp_result" in st.session_state:
        res_exp = st.session_state["exp_result"]
        T_exp = len(res_exp.p1_payoffs)
        rounds_exp = np.arange(1, T_exp + 1)
        actions_mp = ["Heads", "Tails"]

        ex_c1, ex_c2 = st.columns(2)
        with ex_c1:
            # Strategy evolution
            p1_mix_exp = np.array(res_exp.p1_mixed)
            fig_q = go.Figure()
            colors = ["steelblue", "salmon"]
            for i, a in enumerate(actions_mp):
                fig_q.add_trace(go.Scatter(
                    x=rounds_exp, y=p1_mix_exp[:, i], name=a, mode="lines",
                    line=dict(color=colors[i], width=1.5),
                ))
            fig_q.add_hline(y=0.5, line_dash="dash", line_color="green",
                            annotation_text="Nash = 0.5")
            fig_q.update_layout(
                title="P1 strategy evolution", yaxis_range=[0, 1],
                xaxis_title="Round", yaxis_title="Probability",
                height=280, margin=dict(t=40),
            )
            st.plotly_chart(fig_q, use_container_width=True)

        with ex_c2:
            # Regret
            fig_qr = go.Figure()
            fig_qr.add_trace(go.Scatter(
                x=rounds_exp, y=res_exp.p1_regret, name="P1 regret",
                mode="lines", line=dict(color="steelblue", width=2),
            ))
            fig_qr.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_qr.update_layout(
                title="P1 average regret / round",
                xaxis_title="Round", yaxis_title="Avg regret",
                height=280, margin=dict(t=40),
            )
            st.plotly_chart(fig_qr, use_container_width=True)

        st.metric("P1 final avg regret/round", f"{res_exp.p1_regret[-1]:.5f}")
        st.metric("P1 time-average strategy (Heads)", f"{res_exp.p1_avg_mixed[0]:.4f}")


# ============================================================
# THEORY NOTES TAB
# ============================================================
with tab_theory:
    st.subheader("Theory Notes")
    st.markdown("""
## Two-Player Zero-Sum Games

A **zero-sum game** has a payoff matrix A (n×m). Player 1 (row) wants to maximize, Player 2 (column) wants to minimize.

**Nash equilibrium:** a pair (x*, y*) of mixed strategies such that neither player can benefit by unilaterally deviating:
$$x^* \\in \\arg\\max_x x^\\top A y^*, \\quad y^* \\in \\arg\\min_y (x^*)^\\top A y$$

For zero-sum games, the minimax theorem (von Neumann 1928) guarantees:
$$\\max_x \\min_y x^\\top A y = \\min_y \\max_x x^\\top A y = V^*$$

The game value $V^*$ is the expected payoff at Nash equilibrium.

---

## Online Learning & Regret

An algorithm plays a sequence of actions $a_1, \\ldots, a_T$. Its **external regret** against the best fixed action in hindsight is:
$$R_T = \\max_{i} \\sum_{t=1}^T \\ell_i^t - \\sum_{t=1}^T \\ell_{a_t}^t$$

A **no-regret algorithm** has $R_T / T \\to 0$ as $T \\to \\infty$.

**Corollary (Nash convergence):** If both players run no-regret algorithms in a zero-sum game, their time-average strategies converge to Nash equilibrium.

---

## Hedge / Multiplicative Weights Update (MWU)

**Initialization:** $w_i^1 = 1$ for all actions $i$.

**Update:** Given payoff vector $\\ell^t$ (where $\\ell_i^t$ is the payoff if action $i$ were played):
$$w_i^{t+1} = w_i^t \\cdot e^{\\eta \\ell_i^t}, \\quad x^{t+1} = \\frac{w^{t+1}}{\\|w^{t+1}\\|_1}$$

**Regret bound:** $R_T \\leq \\frac{\\ln n}{\\eta} + \\eta T$.
Setting $\\eta = \\sqrt{\\ln n / T}$ gives $R_T = O(\\sqrt{T \\ln n})$.

---

## FTRL — Follow the Regularized Leader

$$x^{t+1} = \\arg\\min_{x \\in \\Delta} \\left[ \\eta \\left\\langle \\sum_{s \\leq t} \\ell^s,\\, x \\right\\rangle + R(x) \\right]$$

| Regularizer R(x) | Solution | Algorithm |
|---|---|---|
| $\\sum_i x_i \\ln x_i$ (entropy) | softmax of cumulative payoffs | = Hedge |
| $\\frac{1}{2}\\|x\\|_2^2$ (L2) | project cumulative payoffs onto simplex | projected gradient |

---

## OMD — Online Mirror Descent

**Dual update:** $\\theta^{t+1} = \\theta^t + \\eta \\ell^t$ (gradient step in dual space)

**Primal recovery:** $x^{t+1} = \\nabla \\Phi^*(\\theta^{t+1})$ via the conjugate of mirror map Φ

| Mirror map Φ | Primal recovery | Algorithm |
|---|---|---|
| $\\sum_i x_i \\ln x_i$ (entropic) | softmax(θ) | = Hedge |
| $\\frac{1}{2}\\|x\\|_2^2$ (Euclidean) | Π_Δ(θ) | projected gradient |

**Relationship:** FTRL and OMD are different algorithms but agree when the mirror map matches the regularizer, and both achieve O(√T) regret.

---

## Fictitious Play

Maintains count of opponent actions. At each round:
1. Compute opponent's empirical frequency $\\hat{y}^t = \\text{counts} / t$
2. Play **best response**: $\\arg\\max_i (A \\hat{y}^t)_i$

**Convergence:** Time-average strategies converge to Nash in two-player zero-sum games (Brown 1951). Instantaneous strategy may cycle. Not a regret minimizer in general.

---

## Key Relationships

```
Hedge  =  OMD with entropic mirror map  =  FTRL with entropy regularizer
            ↕ different algorithms,         ↕ same update rule
Euclidean OMD  =  FTRL with L2 regularizer  =  projected gradient ascent
```

""")
