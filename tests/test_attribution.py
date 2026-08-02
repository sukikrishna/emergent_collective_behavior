"""Attribution stage: do the influence-graph metrics and KCA rank the
ground-truth drivers above the honest agents, on scenarios where we know the
answer?
"""

from __future__ import annotations

import numpy as np

from collectivescope.attribution.influence_graph import InfluenceGraph, kca_attribution
from collectivescope.datasets import make_coalition_scenario, make_collusion_scenario, make_null_scenario
from collectivescope.detection.transfer_entropy import te_matrix_granger


def _top_k_names(scores: np.ndarray, agent_names: list[str], k: int) -> set[str]:
    ranked = sorted(zip(agent_names, scores), key=lambda x: -x[1])
    return {name for name, _ in ranked[:k]}


def test_collusion_drivers_rank_above_honest_agents():
    traj = make_collusion_scenario(
        n_agents=5, n_timesteps=300, shift_step=150, n_drivers=2, coupling=0.9, noise=0.2, seed=1
    )
    W = te_matrix_granger(traj.states, conditional=True)
    ig = InfluenceGraph(W, agent_names=traj.agent_names)
    summary = ig.summary()

    # in_strength, hub_score, and pagerank all account for edges pointing at
    # the follower driver too, so they should recover both true drivers.
    for method in ("in_strength", "hub_score", "pagerank"):
        top2 = _top_k_names(summary[method], traj.agent_names, k=2)
        assert top2 == traj.driver_agents, f"{method} top-2 = {top2}, expected {traj.driver_agents}"

    # out_strength and causal_dominance measure *outgoing* influence, so the
    # leader alone (the dominant source) must be at rank 1 for both.
    for method in ("out_strength", "causal_dominance"):
        top1 = _top_k_names(summary[method], traj.agent_names, k=1)
        assert top1 <= traj.driver_agents, f"{method} top-1 = {top1}, expected subset of {traj.driver_agents}"


def test_kca_ranks_leader_first_in_collusion_scenario():
    traj = make_collusion_scenario(
        n_agents=5, n_timesteps=300, shift_step=150, n_drivers=2, coupling=0.9, noise=0.2, seed=1
    )
    scores = kca_attribution(traj.states, conditional=True)
    top1 = _top_k_names(scores, traj.agent_names, k=1)
    assert top1 <= traj.driver_agents


def test_coalition_drivers_rank_above_honest_agents():
    traj = make_coalition_scenario(n_agents=7, n_timesteps=200, n_drivers=2, coupling=0.8, seed=1)
    W = te_matrix_granger(traj.states, conditional=True)
    ig = InfluenceGraph(W, agent_names=traj.agent_names)
    summary = ig.summary()
    for method in ("out_strength", "hub_score"):
        top1 = _top_k_names(summary[method], traj.agent_names, k=1)
        assert top1 <= traj.driver_agents, f"{method} top-1 = {top1}, expected subset of {traj.driver_agents}"


def test_null_scenario_has_no_dominant_agent():
    # with no real coupling, no single agent's influence should tower over
    # the rest -- out_strength scores should all be roughly comparable.
    traj = make_null_scenario(n_agents=5, n_timesteps=300, seed=1)
    W = te_matrix_granger(traj.states, conditional=True)
    ig = InfluenceGraph(W, agent_names=traj.agent_names)
    out_strength = ig.summary()["out_strength"]
    assert out_strength.max() / (out_strength.mean() + 1e-12) < 3.0
