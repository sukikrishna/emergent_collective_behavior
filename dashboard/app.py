"""CollectiveScope dashboard: interactive walkthrough of all three pipeline
stages (detection, attribution, intervention) on either a synthetic scenario
(no setup required) or a real benchmark run, if DATA_ROOT is set and the
corresponding logs are present.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from collectivescope import run_pipeline
from collectivescope.datasets import (
    colludebench,
    cooperbench,
    make_coalition_scenario,
    make_collusion_scenario,
    make_null_scenario,
    pgg_bench,
    werewolf,
)
from collectivescope.intervention.ablation import ablate_agent, validate_intervention
from collectivescope.viz.plots import plot_attribution_bars, plot_influence_graph, plot_kte_timeline

st.set_page_config(page_title="CollectiveScope", layout="wide")

SYNTHETIC_SCENARIOS = {
    "Collusion (regime shift, 5 agents)": lambda seed: make_collusion_scenario(
        n_agents=5, n_timesteps=300, shift_step=150, n_drivers=2, coupling=0.9, noise=0.2, seed=seed
    ),
    "Coalition (drivers from t=0, 7 agents)": lambda seed: make_coalition_scenario(
        n_agents=7, n_timesteps=200, n_drivers=2, coupling=0.8, seed=seed
    ),
    "Null (no collusion, false-positive check)": lambda seed: make_null_scenario(
        n_agents=5, n_timesteps=300, seed=seed
    ),
}

REAL_ADAPTERS = {
    "ColludeBench": colludebench,
    "Werewolf": werewolf,
    "PGG-Bench": pgg_bench,
    "CooperBench": cooperbench,
}


@st.cache_data(show_spinner=False)
def _load_synthetic(scenario_name: str, seed: int):
    return SYNTHETIC_SCENARIOS[scenario_name](seed)


@st.cache_data(show_spinner=False)
def _load_real(adapter_name: str, run_index: int):
    adapter = REAL_ADAPTERS[adapter_name]
    if adapter_name == "ColludeBench":
        runs = adapter.load_runs(max_runs=run_index + 1)
        return adapter.build_trajectory(runs[run_index])
    if adapter_name == "Werewolf":
        games = adapter.load_games(max_games=run_index + 1)
        return adapter.build_trajectory(games[run_index])
    if adapter_name == "PGG-Bench":
        games = adapter.load_games(max_games=run_index + 1)
        return adapter.build_trajectory(games[run_index])
    if adapter_name == "CooperBench":
        run_dirs = adapter.load_run_dirs()
        return adapter.build_trajectory(run_dirs[run_index])
    raise ValueError(adapter_name)


@st.cache_data(show_spinner="Running detection + attribution...")
def _cached_pipeline(traj, attribution_method: str, n_std: float):
    return run_pipeline(traj, attribution_method=attribution_method, n_std=n_std, run_intervention=False)


st.title("CollectiveScope")
st.caption(
    "Detect the onset of emergent collective behavior, attribute it to the agents driving it, "
    "and causally validate that attribution by intervening."
)

with st.sidebar:
    st.header("Data source")
    source = st.radio("Source", ["Synthetic (no setup)", "Real benchmark"], index=0)

    if source == "Synthetic (no setup)":
        scenario_name = st.selectbox("Scenario", list(SYNTHETIC_SCENARIOS))
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=1, step=1)
        traj = _load_synthetic(scenario_name, int(seed))
    else:
        adapter_name = st.selectbox("Benchmark", list(REAL_ADAPTERS))
        adapter = REAL_ADAPTERS[adapter_name]
        if not adapter.is_available():
            st.warning(
                f"No local data found for {adapter_name}. Set the `DATA_ROOT` environment variable "
                f"to a directory containing `{adapter_name.lower()}/` logs and restart. "
                "See data/README.md for the expected layout. Falling back to a synthetic scenario."
            )
            traj = _load_synthetic("Collusion (regime shift, 5 agents)", 1)
        else:
            run_index = st.number_input("Run index", min_value=0, max_value=999, value=0, step=1)
            try:
                traj = _load_real(adapter_name, int(run_index))
            except (IndexError, FileNotFoundError) as e:
                st.error(f"Could not load run {run_index} for {adapter_name}: {e}")
                st.stop()

    st.header("Detection settings")
    attribution_method = st.selectbox(
        "Primary attribution method", ["kca", "pagerank", "out_strength", "in_strength", "hub_score", "causal_dominance"]
    )
    n_std = st.slider("Detection sensitivity (n_std, lower = more sensitive)", 2.0, 6.0, 4.5, 0.5)

st.subheader("Trajectory")
c1, c2, c3 = st.columns(3)
c1.metric("Agents", traj.n_agents)
c2.metric("Timesteps", traj.n_timesteps)
c3.metric("Ground-truth label", traj.label or "(none)")
if traj.driver_agents:
    st.caption(f"Ground-truth drivers: {sorted(traj.driver_agents)}")
if traj.shift_step is not None:
    st.caption(f"Ground-truth shift step: {traj.shift_step}")

result = _cached_pipeline(traj, attribution_method, float(n_std))

tab_detect, tab_attribute, tab_intervene = st.tabs(["1. Detection", "2. Attribution", "3. Intervention"])

with tab_detect:
    st.markdown("**Kinematic Transfer Entropy** — tracks the *dynamics* of inter-agent information flow "
                "(velocity/acceleration/jerk of the TE series) to flag the onset of a shift, not just its level.")
    onsets = result.onset_timesteps()
    if onsets:
        st.success(f"Detected transition onset(s) at timestep(s): {onsets}")
        if traj.shift_step is not None:
            st.caption(f"vs. ground-truth shift step {traj.shift_step} (delta: {onsets[0] - traj.shift_step:+d})")
    else:
        st.info("No transition detected.")
    fig = plot_kte_timeline(
        result.kte,
        transitions=result.transitions,
        shift_step=(traj.shift_step - result.window) // result.step if traj.shift_step else None,
    )
    st.pyplot(fig)
    plt.close(fig)

with tab_attribute:
    st.markdown("**Influence graph** — who initiates, propagates, or amplifies the collective behavior, "
                "not just an average system-wide score.")
    col_graph, col_bars = st.columns(2)
    static_W = result.te_mats.mean(axis=-1)
    with col_graph:
        fig = plot_influence_graph(static_W, traj.agent_names, driver_agents=traj.driver_agents)
        st.pyplot(fig)
        plt.close(fig)
    with col_bars:
        fig = plot_attribution_bars(result.attribution, traj.agent_names, driver_agents=traj.driver_agents)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Per-method ranking**")
    for method, scores in result.attribution.items():
        ranked = sorted(zip(traj.agent_names, scores), key=lambda x: -x[1])
        st.write(f"`{method}`: " + ", ".join(f"{n} ({v:.3f})" for n, v in ranked))

with tab_intervene:
    st.markdown("**Ablate an agent** and rerun the collective-signal measurement to causally test whether "
                "attribution's top pick actually matters more than a peripheral agent.")

    primary_scores = result.attribution.get(attribution_method, result.attribution["pagerank"])
    default_agent = traj.agent_names[int(np.argmax(primary_scores))]
    col1, col2 = st.columns(2)
    with col1:
        agent_to_ablate = st.selectbox("Agent to ablate", traj.agent_names, index=traj.agent_names.index(default_agent))
    with col2:
        mode = st.selectbox("Ablation mode", ["silence", "shuffle", "remove"])

    if st.button("Run intervention", type="primary"):
        agent_idx = traj.agent_names.index(agent_to_ablate)
        from collectivescope.intervention.ablation import _collective_signal

        baseline = _collective_signal(traj.states, conditional=True)
        ablated_states, _ = ablate_agent(traj.states, agent_idx, mode=mode)
        ablated_signal = _collective_signal(ablated_states, conditional=True)
        drop = (baseline - ablated_signal) / baseline if baseline > 1e-12 else 0.0

        st.metric("Baseline collective signal", f"{baseline:.5f}")
        st.metric(f"After ablating {agent_to_ablate} ({mode})", f"{ablated_signal:.5f}", delta=f"{-drop:.1%}")
        if agent_to_ablate in (traj.driver_agents or set()):
            st.caption(f"{agent_to_ablate} is a ground-truth driver.")

    st.divider()
    st.markdown("**Automatic validation**: ablate the top-attributed agent vs. a control agent "
                "(chosen by lowest total TE degree) and check whether the prediction holds.")
    if st.button("Run automatic validation against a control agent"):
        iv = validate_intervention(traj.states, traj.agent_names, primary_scores, mode="silence", conditional=True)
        st.write(f"Top-attributed agent: **{iv.agent_name}** — signal drop **{iv.signal_drop:.1%}**")
        st.write(f"Control agent: **{iv.control_agent}** — signal drop **{iv.control_drop:.1%}**")
        if iv.validated:
            st.success("Prediction validated: ablating the top-attributed agent hurt the collective signal more.")
        else:
            st.warning("Prediction not validated: the control agent's ablation hurt the signal as much or more.")
