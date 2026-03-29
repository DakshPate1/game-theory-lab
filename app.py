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
    st.subheader("📐 Theory Notes")
    st.caption(
        "A step-by-step learning path. Each section builds on the previous one. "
        "Expand any section to read it — Wikipedia links are provided for every concept."
    )

    with st.expander("1 · What is Game Theory?", expanded=True):
        st.markdown("""
[**Game theory**](https://en.wikipedia.org/wiki/Game_theory) is the mathematical study of strategic
interaction among rational agents. "Strategic" means each player's outcome depends not just on
their own choices, but on the choices of everyone else. "Rational" means each player tries to
maximise their own payoff.

The field was formalised by von Neumann and Morgenstern in *Theory of Games and Economic Behavior*
(1944), and extended by John Nash in 1950 with the concept that now bears his name.

**Two main branches:**
- [**Cooperative game theory**](https://en.wikipedia.org/wiki/Cooperative_game_theory): players can
  form binding agreements. Focuses on how coalitions form and how gains are divided.
- [**Non-cooperative game theory**](https://en.wikipedia.org/wiki/Non-cooperative_game_theory):
  players act independently. Each player optimises for themselves. This is what this lab covers.

**Why does it matter?** Almost every real decision involves other agents: competing firms set
prices, governments design policy, bidders compete in auctions, drivers choose routes. Game theory
gives a language for modelling and reasoning about all of these.
""")

    with st.expander("2 · Two-Player Games and the Bimatrix"):
        st.markdown("""
A [**two-player game**](https://en.wikipedia.org/wiki/Normal-form_game) involves exactly two
decision-makers. Each has a finite set of actions. They choose simultaneously, and each receives
a payoff that depends on the pair of choices.

**Bimatrix representation.** The full specification of a two-player game is a pair of matrices
$(A, B)$ — one payoff matrix per player:

| Symbol | Meaning |
|--------|---------|
| $A[i,j]$ | Player 1's payoff when P1 plays action $i$ and P2 plays action $j$ |
| $B[i,j]$ | Player 2's payoff when P1 plays action $i$ and P2 plays action $j$ |

This pair $(A, B)$ is called a [**bimatrix game**](https://en.wikipedia.org/wiki/Bimatrix_game).

**Zero-sum special case.** If $A + B = 0$ elementwise (i.e. $B = -A$), the game is
[**zero-sum**](https://en.wikipedia.org/wiki/Zero-sum_game): every dollar P1 wins is a dollar P2
loses. Only one matrix is needed. Rock Paper Scissors and Matching Pennies are zero-sum.

**General-sum games** allow mutual gain or mutual loss. Prisoner's Dilemma and Battle of the Sexes
are general-sum. The theory is richer and harder: Nash convergence no longer follows automatically
from no-regret play.

In this lab the payoff matrices are shown in the "Show payoff matrix" expander on the Sandbox tab.
""")

    with st.expander("3 · Normal-Form Games"):
        st.markdown("""
The [**normal form**](https://en.wikipedia.org/wiki/Normal-form_game) (also called strategic form)
is the standard way to write down a finite game. It lists:

1. The **players** (here: P1 and P2)
2. Each player's **action set** (finite list of choices)
3. A **payoff function** mapping every action profile $(i, j)$ to payoffs $(A[i,j], B[i,j])$

This representation captures *simultaneous* games — neither player knows what the other will
choose when they act. Both choose at the same moment (or equivalently, in isolation without
communication).

**Contrast: extensive form.** The [extensive form](https://en.wikipedia.org/wiki/Extensive-form_game)
represents *sequential* games — player 2 moves after seeing player 1's choice (like chess). This
lab only uses the normal form.

**Key assumption: common knowledge of rationality.** Both players know the payoff matrices, know
that the other player is rational, know that the other player knows this, and so on. This is what
makes Nash equilibrium the natural solution concept.

**Examples in this lab:**

| Game | Actions | Type | Feature |
|------|---------|------|---------|
| Rock Paper Scissors | 3 × 3 | Zero-sum | Uniform Nash |
| Matching Pennies | 2 × 2 | Zero-sum | Simplest non-trivial |
| Biased RPS | 3 × 3 | Zero-sum | Non-uniform Nash |
| Prisoner's Dilemma | 2 × 2 | General-sum | Dominant strategy |
| Battle of the Sexes | 2 × 2 | General-sum | Multiple Nash equilibria |
""")

    with st.expander("4 · Pure and Mixed Strategies"):
        st.markdown("""
A [**pure strategy**](https://en.wikipedia.org/wiki/Strategy_(game_theory)#Pure_strategy) is a
deterministic choice: always play action $i$. Simple, but exploitable — if P2 knows P1 always
plays Rock, P2 can always play Paper.

A [**mixed strategy**](https://en.wikipedia.org/wiki/Strategy_(game_theory)#Mixed_strategy) is a
probability distribution over actions: play action $i$ with probability $x_i$. Written as a vector
$x \\in \\Delta^n$ where $\\Delta^n = \\{x \\geq 0 : \\sum_i x_i = 1\\}$ is the
[**probability simplex**](https://en.wikipedia.org/wiki/Simplex#Probability_distribution_as_simplex).

**Why mix?** Randomisation makes you *unpredictable*. In RPS, any pure strategy loses to a fixed
counter-strategy. The unique Nash equilibrium is uniform $(1/3, 1/3, 1/3)$ — no opponent can
exploit you if you play each action equally.

**Expected payoff.** When P1 plays mixed strategy $x$ and P2 plays $y$, P1's expected payoff is:
$$u_1(x, y) = x^\\top A y = \\sum_{i,j} x_i \\, A[i,j] \\, y_j$$

This is multilinear in $(x, y)$: linear in $x$ for fixed $y$, and linear in $y$ for fixed $x$.
That linearity is what makes minimax theorems and LP-based Nash computation possible.

**Support of a mixed strategy.** The
[**support**](https://en.wikipedia.org/wiki/Support_(mathematics)) of $x$ is the set of actions
played with positive probability: $\\text{supp}(x) = \\{i : x_i > 0\\}$. At Nash equilibrium,
every action in the support must yield the same expected payoff — otherwise the player would shift
weight to the better action.
""")

    with st.expander("5 · Nash Equilibrium"):
        st.markdown("""
A [**Nash equilibrium**](https://en.wikipedia.org/wiki/Nash_equilibrium) is a pair of mixed
strategies $(x^\\ast, y^\\ast)$ such that neither player can increase their expected payoff by
deviating unilaterally:

$$x^\\ast \\in \\arg\\max_{x \\in \\Delta} \\; x^\\top A y^\\ast \\qquad \\text{and} \\qquad
y^\\ast \\in \\arg\\max_{y \\in \\Delta} \\; y^\\top B^\\top x^\\ast$$

Each player is playing a best response to the other. No one has an incentive to change.

**Existence.** [Nash's theorem (1950)](https://en.wikipedia.org/wiki/Nash_equilibrium#Proof_of_existence)
guarantees that every finite game has at least one Nash equilibrium in mixed strategies. This is
proven via Kakutani's fixed point theorem applied to the best-response correspondence.

**Zero-sum games: Minimax Theorem.** Von Neumann (1928) showed:
$$\\max_{x \\in \\Delta} \\min_{y \\in \\Delta} x^\\top A y \\;=\\; \\min_{y \\in \\Delta} \\max_{x \\in \\Delta} x^\\top A y \\;=\\; V^\\ast$$

$V^\\ast$ is the **game value** — the expected payoff P1 can guarantee regardless of P2's strategy.
The Nash equilibrium strategies are exactly the minimax/maximin strategies.

**Computing Nash.** For zero-sum games: a
[**linear program**](https://en.wikipedia.org/wiki/Linear_programming). For general-sum games:
[**Lemke–Howson algorithm**](https://en.wikipedia.org/wiki/Lemke%E2%80%93Howson_algorithm) or
support enumeration — both are in PPAD (not known to be polynomial in the worst case).

**Uniqueness.** Nash equilibria are not unique in general. Prisoner's Dilemma has one (Defect,
Defect). Battle of the Sexes has three (two pure, one mixed). For zero-sum games the Nash strategies
may differ but all share the same game value $V^\\ast$.
""")

    with st.expander("6 · Why Do Algorithms Exist? The Online Learning Framing"):
        st.markdown("""
Computing Nash equilibrium exactly requires knowing both players' payoff matrices fully — and is
computationally hard for general-sum games. In practice, you often face a different situation:

- You don't fully know the opponent's payoffs
- You interact repeatedly and observe outcomes
- You want to do well *in the long run* without solving a global optimisation first

This is the setting of [**online learning**](https://en.wikipedia.org/wiki/Online_machine_learning).

**The protocol.** For $t = 1, 2, \\ldots, T$:
1. You choose a mixed strategy $x^t \\in \\Delta$
2. The environment (opponent) reveals their action or your loss vector $\\ell^t$
3. You update your strategy for the next round

No assumptions on the opponent — they can be adversarial, adaptive, or random.

**The goal.** Minimise regret — the gap between your cumulative payoff and what you would have
earned with the best *fixed* strategy in hindsight.

**The payoff.** If both players do this simultaneously, something remarkable happens: their
time-average strategies converge to Nash equilibrium, even though neither player was directly
trying to compute Nash. The equilibrium emerges from the interaction.

This is why the algorithms matter — they give you a practical, feedback-driven path to Nash
without needing full information about the game.
""")

    with st.expander("7 · Regret: The Core Metric"):
        st.markdown("""
[**Regret**](https://en.wikipedia.org/wiki/Regret_(decision_theory)) measures how much worse
you did compared to the best strategy you *could* have used, knowing the full history.

**External regret** (the standard definition):
$$R_T = \\max_{i \\in \\text{actions}} \\sum_{t=1}^T \\ell_i^t \\;-\\; \\sum_{t=1}^T \\ell_{a_t}^t$$

Where $\\ell_i^t$ is the payoff of action $i$ at round $t$, and $a_t$ is the action actually played.

The first term is the cumulative payoff of the best *fixed* action in hindsight.
The second is your actual cumulative payoff.
The difference is how much you "regret" not having committed to that best action from the start.

**Average regret** is $R_T / T$. A **no-regret algorithm** has $R_T / T \\to 0$.

**Intuition.** Regret is not about being sad — it's a formal measure of adaptiveness. An algorithm
with zero average regret is doing as well as the best fixed strategy in any environment. That's a
strong guarantee.

**What external regret does NOT capture:**
- [**Internal regret**](https://en.wikipedia.org/wiki/Regret_(decision_theory)#Internal_regret):
  "What if I had swapped action $i$ for action $j$ every time I played $i$?" Minimising internal
  regret leads to [correlated equilibrium](https://en.wikipedia.org/wiki/Correlated_equilibrium).
- [**Swap regret**](https://en.wikipedia.org/wiki/Swap_regret): generalises internal regret;
  also leads to correlated equilibrium.

**Nash regret.** Sometimes written as the Nash gap — how far the time-average strategies are from
Nash equilibrium. Formally: if $(\\bar{x}, \\bar{y})$ are time-averages, Nash regret is:
$$\\text{NashGap}(\\bar{x}, \\bar{y}) = \\max_{x} x^\\top A \\bar{y} \\;-\\; \\min_{y} \\bar{x}^\\top A y$$

This is what the **Nash Gap / Exploitability** plot shows. It is bounded by
$(R_1(T) + R_2(T)) / T$, so as both players' regret drops, the Nash gap shrinks too.
""")

    with st.expander("8 · Exploitability"):
        st.markdown("""
[**Exploitability**](https://en.wikipedia.org/wiki/Exploitability) measures how much a player's
strategy can be taken advantage of by an optimal opponent.

For a single player with mixed strategy $x$, exploitability is:
$$\\text{Exploit}_1(x) = \\max_{i} (Ay^\\ast)_i - x^\\top A y^\\ast$$

In practice (without knowing $y^\\ast$), we compute it differently: given the opponent's
*current* strategy $y$, how much can a best-responding P1 gain beyond what $x$ achieves?

$$\\text{Exploit}_1(x, y) = \\max_i (Ay)_i - x^\\top A y$$

This is what the **Nash Gap** tab plots. Zero exploitability means $x$ is already a best response
to $y$ — the pair is at Nash equilibrium.

**Exploitability of Nash = 0.** By definition, no player can gain by deviating from Nash. So the
Nash strategy is unexploitable.

**Exploitability of uniform in RPS = 0.** Against uniform, every action has expected payoff 0
regardless. No opponent can exploit uniform. This is why **Uniform vs Adversarial BR** gives the
adversary no edge — uniform is already Nash for RPS.

**Practical note.** In poker and other large games, exploitability is the standard metric for
evaluating AI agents. A fully unexploitable agent plays Nash. A low-exploitability agent is nearly
Nash — safe against any opponent.
""")

    with st.expander("9 · Convergence"):
        st.markdown("""
[**Convergence**](https://en.wikipedia.org/wiki/Convergence_(mathematics)) in this context means
the strategies or payoffs approach a stable limit as the number of rounds grows.

There are two distinct things that converge:

**1. Time-average strategy convergence.**
The time-average $\\bar{x}^T = \\frac{1}{T} \\sum_{t=1}^T x^t$ converges to Nash equilibrium
when both players run no-regret algorithms in a zero-sum game. This is the convergence you see in
the *Strategy Evolution* plot when averaging over rounds.

The instantaneous strategy $x^t$ itself may oscillate or cycle and need not converge.

**2. Regret convergence.**
$R_T / T \\to 0$. This is the regret plot. Sublinear regret means the algorithm is learning —
the per-round gap to best fixed action shrinks toward zero.

**Rate of convergence.** How fast does $R_T / T$ shrink? Hedge achieves
$R_T / T = O(\\sqrt{\\ln n / T})$. This means:
- Double the rounds → regret per round shrinks by $\\sqrt{2}$
- The regret curve in the plot should look like $1/\\sqrt{T}$

**Lower bound on convergence.** No online algorithm can achieve $R_T = o(\\sqrt{T})$ in the
worst case (Cesa-Bianchi & Lugosi 2006). The $O(\\sqrt{T})$ rate is *optimal* — you cannot do
better in a fully adversarial environment.

**Convergence vs Nash.** These are linked but different:
- Low regret ⟹ time-average near Nash (in zero-sum)
- Nash ⟹ zero exploitability
- Exploitability shrinks at rate $O(1/\\sqrt{T})$ when both use Hedge
""")

    with st.expander("10 · The Algorithms"):
        st.markdown("""
All algorithms in this lab solve the same problem: choose mixed strategies online to minimise
regret. They differ in *how* they use past payoff information.

---

**Hedge / Multiplicative Weights Update (MWU)**
[Wikipedia](https://en.wikipedia.org/wiki/Multiplicative_weights_update_method)

The canonical no-regret algorithm. Maintains a weight $w_i$ per action; after each round,
multiplies the weight of each action by $e^{\\eta \\ell_i^t}$ (exponential in payoff), then
normalises. Actions that pay off get exponentially more weight.

$$w_i^{t+1} = w_i^t \\cdot e^{\\eta \\ell_i^t}, \\qquad x^{t+1} = \\frac{w^{t+1}}{\\|w^{t+1}\\|_1}$$

**Regret bound:** $R_T \\leq \\frac{\\ln n}{\\eta} + \\eta T$. Optimal $\\eta = \\sqrt{\\ln n / T}$
gives $R_T = O(\\sqrt{T \\ln n})$.

---

**FTRL — Follow the Regularized Leader**
[Wikipedia](https://en.wikipedia.org/wiki/Follow_the_regularized_leader)

At each round, choose the strategy that minimises cumulative past loss plus a regularisation term:
$$x^{t+1} = \\arg\\min_{x \\in \\Delta} \\left[ \\eta \\left\\langle \\sum_{s \\leq t} \\ell^s, x \\right\\rangle + R(x) \\right]$$

Regularisation prevents wild swings. With entropy regulariser → Hedge. With L2 regulariser →
projected gradient descent. The regulariser is a design choice.

---

**OMD — Online Mirror Descent**
[Wikipedia](https://en.wikipedia.org/wiki/Mirror_descent)

Takes a gradient step in a *dual space* and maps back to the simplex via a mirror map $\\Phi$:
$$\\theta^{t+1} = \\theta^t + \\eta \\ell^t \\quad \\text{(dual step)}, \\qquad
x^{t+1} = \\nabla \\Phi^\\ast(\\theta^{t+1}) \\quad \\text{(primal recovery)}$$

With entropic mirror map → Hedge. With Euclidean mirror map → projected gradient. Same regret
bounds as FTRL; different derivation.

---

**Fictitious Play**
[Wikipedia](https://en.wikipedia.org/wiki/Fictitious_play)

Tracks opponent's empirical action frequency $\\hat{y}^t$ and best-responds to it. Older (Brown
1951) and not a regret minimiser in general, but time-average converges to Nash in zero-sum games.

---

**Equivalences:**
```
Hedge  =  OMD (entropic)  =  FTRL (entropy regulariser)
Euclidean OMD  =  FTRL (L2 regulariser)  =  projected gradient
```
""")

    with st.expander("11 · Regret Bounds and Why They Exist"):
        st.markdown("""
**Why is $O(\\sqrt{T})$ the right rate?**

The bound $R_T = O(\\sqrt{T \\ln n})$ comes from a bias-variance tradeoff in the learning rate:

- Large $\\eta$: weights shift fast → good at exploiting payoff differences, but overshoots and
  oscillates. Regret from instability: $O(\\eta T)$.
- Small $\\eta$: weights shift slowly → stable, but slow to adapt. Regret from slow learning:
  $O(\\ln n / \\eta)$.

Setting $\\eta = \\sqrt{\\ln n / T}$ balances these two terms, giving $R_T = 2\\sqrt{T \\ln n}$.

**The lower bound.** For any online algorithm in an adversarial environment, there exists a loss
sequence such that $R_T = \\Omega(\\sqrt{T})$. The $\\sqrt{T}$ rate is tight — no algorithm can
guarantee sub-$\\sqrt{T}$ regret against an adversary. The $\\ln n$ factor reflects: more actions
= more uncertainty = higher regret, but only logarithmically (information-theoretic reason).

**Why $\\ln n$ not $n$?** Because the algorithm only needs to identify *which* action is best,
not learn its exact payoff. That requires $O(\\ln n)$ "bits" of information, not $O(n)$.

**Practical consequence.** For RPS ($n=3$) over $T=1000$ rounds:
$$\\frac{R_T}{T} \\leq \\sqrt{\\frac{\\ln 3}{1000}} \\approx 0.033$$

This means Hedge will be within 0.033 per round of the best fixed action — very tight already.
At $T=10{,}000$: $\\approx 0.010$. The bound converges even for small games.
""")

    with st.expander("12 · Real-World Applications"):
        st.markdown("""
Game theory and no-regret algorithms are not academic exercises. Here are concrete deployments:

**Auctions and Market Design**
[Auction theory](https://en.wikipedia.org/wiki/Auction_theory) is one of the most applied areas
of game theory. Google and Microsoft run billions of
[**ad auctions**](https://en.wikipedia.org/wiki/Generalized_second-price_auction) per day where
advertisers bid strategically. Spectrum auctions (FCC) use game-theoretic mechanism design to
allocate radio frequencies efficiently. No-regret bidding strategies ensure buyers don't get
systematically exploited.

**Traffic and Routing**
[**Selfish routing**](https://en.wikipedia.org/wiki/Selfish_routing) models drivers choosing
routes to minimise travel time. Each driver is a player; roads are shared resources. Nash
equilibrium in this game can be socially *worse* than the optimal assignment (Braess's paradox).
Online algorithms are used to route packets in networks and assign jobs to servers.

**Security Games**
[**Stackelberg security games**](https://en.wikipedia.org/wiki/Stackelberg_competition) model
situations where a defender (e.g. airport security, police patrols) allocates resources against
an attacker who observes the pattern and exploits gaps. The ARMOR system deployed at LAX (2007)
used game-theoretic scheduling to randomise security checkpoints.

**Economics and Finance**
[**Algorithmic game theory**](https://en.wikipedia.org/wiki/Algorithmic_game_theory) analyses
market equilibria computationally. Hedge-style algorithms are used in
[**online portfolio selection**](https://en.wikipedia.org/wiki/Universal_portfolio_algorithm)
(Cover 1991) — the multiplicative weights algorithm achieves the same return as the best fixed
portfolio allocation in hindsight.

**Machine Learning**
[**Generative Adversarial Networks (GANs)**](https://en.wikipedia.org/wiki/Generative_adversarial_network)
are literally a two-player zero-sum game: a generator vs a discriminator. Training is finding Nash
equilibrium of this game. Many GAN failure modes (mode collapse, oscillation) are exactly the
phenomena you can observe here when algorithms don't converge.

**Multi-Agent Reinforcement Learning**
[**MARL**](https://en.wikipedia.org/wiki/Multi-agent_reinforcement_learning) is the broader field
where multiple RL agents interact. No-regret algorithms are the foundation of self-play training —
AlphaGo/AlphaZero and poker AIs (Libratus, Pluribus) use variants of CFR (Counterfactual Regret
Minimisation), which is regret minimisation in extensive-form games.
""")

    with st.expander("13 · Devising a Strategy: What to Think About"):
        st.markdown("""
When choosing an algorithm or designing a strategy for a two-player game, consider:

**Know your game structure**
- Zero-sum or general-sum? No-regret algorithms converge to Nash only in zero-sum. In general-sum,
  you need different solution concepts (correlated equilibrium, social welfare maximisation).
- How many actions does each player have? The $\\ln n$ term in regret bounds grows slowly — even
  100 actions only adds $\\ln 100 \\approx 4.6$ compared to $\\ln 2 = 0.69$.
- Is the game repeated? Regret minimisation applies to *repeated* play. Single-shot games are
  different.

**Know your horizon**
- Fixed $T$: set $\\eta = \\sqrt{\\ln n / T}$ to get optimal regret. If you don't know $T$, use
  [**doubling trick**](https://en.wikipedia.org/wiki/Doubling_trick) or adaptive learning rates.
- Infinite horizon: use $\\eta_t = 1/\\sqrt{t}$ (decreasing learning rate). Regret still sublinear.

**Know your opponent**
- Stochastic opponent (fixed distribution): you can learn faster; Hedge still works but UCB-style
  algorithms can achieve $O(\\ln T)$ regret (logarithmic, not $\\sqrt{T}$).
- Adversarial opponent: you need Hedge-style algorithms with the full $O(\\sqrt{T})$ bound.
- Adaptive opponent (changes based on your play): the Stackelberg adversary in this lab. Hedge's
  bound still holds.

**What regulariser to use**
- Entropy (Hedge): keeps all actions in play (strictly positive probability). Good when you want
  full exploration. Slightly more stable numerically.
- L2 (Euclidean OMD): can concentrate on fewer actions. Better when the game has a sparse Nash
  (pure strategy or near-pure). Can be more aggressive.

**Learning rate sensitivity**
- Too large: fast initial adaptation, but the strategy oscillates and regret per round doesn't
  decay well. Watch for this in the "Strategy Evolution" plot.
- Too small: slow to adapt, loses to a changing opponent in early rounds. Regret initially large.
- Optimal: use the formula, or tune with the slider and watch the regret plot flatten.

**When NOT to use no-regret algorithms**
- When you can solve the game exactly (small zero-sum game + known payoffs → just use LP).
- When the game is one-shot (no learning possible).
- When you care about [correlated equilibrium](https://en.wikipedia.org/wiki/Correlated_equilibrium)
  instead of Nash (use swap regret minimisation instead).
""")

    with st.expander("14 · Glossary: Every Term Defined"):
        st.markdown("""
**Action** — A single choice available to a player in one round. In RPS: Rock, Paper, Scissors.

**Action profile** — The tuple of actions chosen by all players simultaneously: $(i, j)$ for
two players.

**Adversarial opponent** — An opponent who can choose loss sequences to maximise your regret.
The regret bounds for Hedge hold even against this worst case.

**Average regret** — $R_T / T$: regret per round. Converges to 0 for no-regret algorithms.

**Best response** — The action (or mixed strategy) that maximises your expected payoff given the
opponent's strategy. Denoted $\\text{BR}(y) = \\arg\\max_i (Ay)_i$.

**Bimatrix game** — A two-player game represented by two payoff matrices $(A, B)$, one per player.

**Convergence** — The time-average strategies approaching Nash equilibrium as $T \\to \\infty$.

**Correlated equilibrium** — A generalisation of Nash where players coordinate via a shared
signal. A distribution over action profiles where no player wants to deviate given the signal.
Achievable via *swap regret* minimisation. Every Nash equilibrium is a correlated equilibrium
but not vice versa.

**Dominant strategy** — An action that is a best response to *every* opponent strategy. In
Prisoner's Dilemma, Defect dominates Cooperate regardless of what the opponent does.

**Exploitability** — How much a player's strategy can be improved against a best-responding
opponent. Zero exploitability ⟺ Nash equilibrium.

**External regret** — The standard regret definition: cumulative payoff of best fixed action in
hindsight minus your actual cumulative payoff.

**Extensive form** — A representation of sequential games using a game tree, where players move
one at a time with possible information asymmetry.

**Fictitious Play** — A learning algorithm that best-responds to the opponent's empirical
frequency. Converges in time-average for zero-sum games; not a regret minimiser.

**Game value** ($V^\\ast$) — In a zero-sum game: the expected payoff at Nash equilibrium. P1
can guarantee at least $V^\\ast$; P2 can hold P1 to at most $V^\\ast$.

**Hedge / MWU** — Multiplicative Weights Update. The canonical no-regret algorithm for the
simplex. Identical to entropic OMD and FTRL with entropy regulariser.

**Internal regret** — "If I had swapped every action $i$ for action $j$, how much would I have
gained?" Minimising internal regret leads to correlated equilibrium.

**Learning rate ($\\eta$)** — Controls how aggressively the algorithm responds to new payoff
information. Optimal value: $\\eta = \\sqrt{\\ln n / T}$.

**Minimax theorem** — Von Neumann (1928): in any finite two-player zero-sum game,
$\\max_x \\min_y x^\\top A y = \\min_y \\max_x x^\\top A y = V^\\ast$.

**Mixed strategy** — A probability distribution $x \\in \\Delta$ over actions.

**Nash equilibrium** — A pair of strategies where neither player can gain by unilaterally
deviating. Every finite game has at least one (Nash 1950).

**Nash gap** — $\\max_x x^\\top A \\bar{y} - \\min_y \\bar{x}^\\top A y$: how far
time-average strategies are from Nash. Bounded by $(R_1 + R_2)/T$.

**No-regret algorithm** — Any algorithm with $R_T / T \\to 0$. Examples: Hedge, FTRL, OMD.

**Normal form** — The payoff-matrix representation of a simultaneous-move game.

**Online Mirror Descent (OMD)** — A family of no-regret algorithms that take gradient steps in a
dual space and map back via a mirror map.

**FTRL (Follow the Regularized Leader)** — Chooses the strategy minimising cumulative past loss
plus a regulariser. Hedge and projected gradient are special cases.

**Payoff matrix** ($A$) — The $n \\times m$ matrix where $A[i,j]$ is Player 1's payoff when P1
plays action $i$ and P2 plays action $j$.

**Probability simplex** ($\\Delta$) — The set of all valid mixed strategies:
$\\Delta = \\{x \\in \\mathbb{R}^n : x_i \\geq 0, \\sum_i x_i = 1\\}$.

**Pure strategy** — A deterministic action choice: always play action $i$.

**Regret** — Cumulative loss compared to the best fixed action in hindsight.

**Regret bound** — A worst-case upper bound on regret: Hedge achieves $R_T \\leq 2\\sqrt{T \\ln n}$.

**Saddle point** — A strategy pair $(x^\\ast, y^\\ast)$ satisfying
$x^\\top A y^\\ast \\leq (x^\\ast)^\\top A y^\\ast \\leq (x^\\ast)^\\top A y$ for all $x, y$.
In zero-sum games, saddle points = Nash equilibria.

**Stackelberg adversary** — An opponent who observes your mixed strategy before choosing. The
theoretical worst case for a learner; Hedge's regret bound still holds against this.

**Support** — The set of actions played with positive probability in a mixed strategy.

**Time-average strategy** ($\\bar{x}^T$) — The average of all mixed strategies played up to
round $T$: $\\bar{x}^T = \\frac{1}{T} \\sum_{t=1}^T x^t$. Converges to Nash for no-regret
algorithms in zero-sum games.

**Zero-sum game** — A game where $A + B = 0$: one player's gain is exactly the other's loss.
""")

    with st.expander("15 · Key Relationships Cheat Sheet"):
        st.markdown("""
```
Hedge  =  OMD (entropic mirror map)  =  FTRL (entropy regulariser)
           ↕ same update, 3 derivations

Euclidean OMD  =  FTRL (L2 regulariser)  =  projected gradient descent
           ↕ same update, 3 derivations

Both families: O(√(T ln n)) regret — optimal in the adversarial setting
```

**When both players use no-regret algorithms (zero-sum):**
```
R₁(T)/T → 0  and  R₂(T)/T → 0
        ⟹
Nash gap → 0  (time-average strategies converge to Nash)
```

**Hierarchy of equilibrium concepts (easiest to hardest to compute):**
```
Correlated equilibrium  ⊇  Coarse correlated equilibrium
        ↑ swap regret minimisation achieves this
Nash equilibrium  ⊆  Correlated equilibrium
        ↑ external regret minimisation achieves Nash (in zero-sum)
```

**Regret bound summary:**

| Algorithm | Regret bound | Notes |
|-----------|-------------|-------|
| Hedge (optimal η) | $2\\sqrt{T \\ln n}$ | Optimal rate |
| FTRL (entropy) | $2\\sqrt{T \\ln n}$ | Identical to Hedge |
| OMD (entropic) | $2\\sqrt{T \\ln n}$ | Identical to Hedge |
| OMD / FTRL (L2) | $O(\\sqrt{T})$ | Constant depends on geometry |
| Fictitious Play | Not bounded | Not a regret minimiser |
| Best Response (to last) | Not bounded | Can have linear regret |
| Uniform | $O(T)$ worst case | No adaptation |
""")

