"""Intervention stage: does ablating the top-attributed agent hurt the
collective signal more than ablating a genuine control, closing the
observe -> intervene loop?

test_control_agent_is_not_the_other_driver is a regression guard for a real
bug found during development: validate_intervention() originally picked the
control agent as argmin(attribution_scores). For an outgoing-influence score
like kca/out_strength, a *follower* driver can score lowest while still
sitting on the graph's single highest-weight edge (as the target of the
leader) -- silencing it then collapsed the collective signal even more than
silencing the true leader, purely because it shares the leader's edge, and
the intervention wrongly reported "prediction not validated" on every
synthetic scenario. The fix selects the control by total TE degree
(in-strength + out-strength) instead, which is what "peripheral agent"
actually means here.
"""

from __future__ import annotations

from collectivescope.attribution.influence_graph import kca_attribution
from collectivescope.datasets import make_coalition_scenario, make_collusion_scenario
from collectivescope.intervention.ablation import ablate_agent, validate_intervention
from collectivescope.pipeline import run_pipeline


def test_ablating_leader_validates_in_collusion_scenario():
    traj = make_collusion_scenario(
        n_agents=5, n_timesteps=300, shift_step=150, n_drivers=2, coupling=0.9, noise=0.2, seed=1
    )
    scores = kca_attribution(traj.states, conditional=True)
    result = validate_intervention(traj.states, traj.agent_names, scores, mode="silence", conditional=True)
    assert result.agent_name in traj.driver_agents
    assert result.control_agent not in traj.driver_agents
    assert result.validated


def test_control_agent_is_not_the_other_driver():
    traj = make_collusion_scenario(
        n_agents=5, n_timesteps=300, shift_step=150, n_drivers=2, coupling=0.9, noise=0.2, seed=1
    )
    scores = kca_attribution(traj.states, conditional=True)
    result = validate_intervention(traj.states, traj.agent_names, scores, mode="silence", conditional=True)
    # the control must be a genuinely peripheral agent, not the follower half
    # of the true driver pair (which sits on the same dominant edge).
    assert result.control_agent not in traj.driver_agents


def test_ablating_leader_validates_in_coalition_scenario():
    traj = make_coalition_scenario(n_agents=7, n_timesteps=200, n_drivers=2, coupling=0.8, seed=1)
    scores = kca_attribution(traj.states, conditional=True)
    result = validate_intervention(traj.states, traj.agent_names, scores, mode="silence", conditional=True)
    assert result.agent_name in traj.driver_agents
    assert result.validated


def test_full_pipeline_runs_intervention_end_to_end():
    traj = make_collusion_scenario(
        n_agents=5, n_timesteps=300, shift_step=150, n_drivers=2, coupling=0.9, noise=0.2, seed=1
    )
    result = run_pipeline(traj, run_intervention=True)
    assert result.intervention is not None
    assert result.intervention.validated


def test_ablate_agent_modes_preserve_shape_or_drop_one_agent():
    traj = make_collusion_scenario(n_agents=5, n_timesteps=50, seed=0)
    for mode in ("silence", "shuffle"):
        new_traj, n_agents = ablate_agent(traj.states, 0, mode=mode)
        assert new_traj.shape == traj.states.shape
        assert n_agents == traj.n_agents
    removed_traj, n_agents = ablate_agent(traj.states, 0, mode="remove")
    assert n_agents == traj.n_agents - 1
    assert removed_traj.shape[1] == traj.n_agents - 1
