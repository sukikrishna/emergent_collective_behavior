"""Intervention: act on the agents attribution identified as influential, then
rerun detection + attribution and check whether the collective signal moves
the way the attribution predicted.

This is the piece of the pipeline that turns attribution from a descriptive
map into a causally-tested claim: "agent X is a driver" is falsifiable —
removing/silencing/shuffling X should collapse the collective signal more
than doing the same to a non-driver. ``validate_intervention`` runs exactly
that comparison and reports whether the prediction held.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detection.transfer_entropy import te_matrix_granger


def ablate_agent(traj: np.ndarray, agent_idx: int, mode: str = "silence") -> tuple[np.ndarray, int]:
    """Remove, silence, or shuffle one agent's contribution to a continuous
    trajectory (T, n_agents, d).

    mode:
      "remove"  — drop the agent's column entirely (returns n_agents - 1).
      "silence" — replace the agent's trajectory with its own time-mean, i.e.
                  simulate "the agent stops acting" without changing the
                  agent count (a constant series carries no information, so
                  its outgoing TE to everyone else should collapse).
      "shuffle" — replace the agent's trajectory with a temporally shuffled
                  copy of itself: same marginal statistics, but the temporal
                  structure that TE depends on is destroyed.

    Returns (new_traj, new_n_agents).
    """
    T, n_agents, d = traj.shape
    if mode == "remove":
        keep = [a for a in range(n_agents) if a != agent_idx]
        return traj[:, keep, :], len(keep)
    if mode == "silence":
        t2 = traj.copy()
        t2[:, agent_idx, :] = traj[:, agent_idx, :].mean(axis=0, keepdims=True)
        return t2, n_agents
    if mode == "shuffle":
        t2 = traj.copy()
        perm = np.random.default_rng(0).permutation(T)
        t2[:, agent_idx, :] = traj[perm, agent_idx, :]
        return t2, n_agents
    raise ValueError(f"unknown ablation mode: {mode!r}")


def _collective_signal(traj: np.ndarray, conditional: bool = True) -> float:
    """Scalar collective-influence signal used as the intervention metric: mean
    conditional TE over every directed off-diagonal pair. This is the same
    quantity the detection stage averages over time into KTE — here we just
    need one number per (ablated) trajectory to compare conditions."""
    n_agents = traj.shape[1]
    if n_agents < 2:
        return 0.0
    W = te_matrix_granger(traj, conditional=conditional)
    off_diag = ~np.eye(n_agents, dtype=bool)
    return float(W[off_diag].mean())


@dataclass
class InterventionResult:
    agent_name: str
    mode: str
    baseline_signal: float
    ablated_signal: float
    control_signal: float
    control_agent: str

    @property
    def signal_drop(self) -> float:
        """Fraction of the baseline collective signal lost by ablating this
        agent. Positive means the signal fell, as predicted for a real driver."""
        if self.baseline_signal < 1e-12:
            return 0.0
        return (self.baseline_signal - self.ablated_signal) / self.baseline_signal

    @property
    def control_drop(self) -> float:
        if self.baseline_signal < 1e-12:
            return 0.0
        return (self.baseline_signal - self.control_signal) / self.baseline_signal

    @property
    def validated(self) -> bool:
        """The attribution's causal prediction holds if ablating the
        attributed agent hurts the collective signal more than ablating the
        control agent did."""
        return self.signal_drop > self.control_drop


def validate_intervention(
    traj: np.ndarray,
    agent_names: list[str],
    attribution_scores: np.ndarray,
    mode: str = "silence",
    conditional: bool = True,
) -> InterventionResult:
    """Close the observe -> intervene loop for one trajectory.

    Takes the top-attributed agent (highest ``attribution_scores``) and
    ablates it, then compares the resulting collective-signal drop against
    ablating a control agent. If attribution is causally meaningful, ablating
    the top driver should hurt the collective signal more than ablating the
    control.

    The control is chosen by *total TE degree* (in-strength + out-strength
    from a fresh full TE matrix), not by ``attribution_scores`` directly.
    Picking the argmin of ``attribution_scores`` is confounded by network
    topology: e.g. a follower agent can have the *lowest* outgoing-influence
    score (kca/out-strength) while still sitting on the single
    highest-weight edge in the graph (as the *target* of the top driver).
    Silencing either endpoint of that edge collapses it from the off-diagonal
    mean, so an argmin-by-outgoing-score control can show a bigger signal
    drop than the true driver purely because it shares the driver's edge —
    not because it's peripheral. Total-degree selects an agent that is
    actually weakly connected in both directions, which is what "control"
    is supposed to mean here.
    """
    top_idx = int(np.argmax(attribution_scores))

    W = te_matrix_granger(traj, conditional=conditional)
    total_degree = W.sum(axis=0) + W.sum(axis=1)  # in-strength + out-strength per agent
    order = np.argsort(total_degree)
    control_idx = int(order[0])
    if control_idx == top_idx and len(order) > 1:
        control_idx = int(order[1])

    baseline = _collective_signal(traj, conditional=conditional)

    ablated_traj, _ = ablate_agent(traj, top_idx, mode=mode)
    ablated_signal = _collective_signal(ablated_traj, conditional=conditional)

    control_traj, _ = ablate_agent(traj, control_idx, mode=mode)
    control_signal = _collective_signal(control_traj, conditional=conditional)

    return InterventionResult(
        agent_name=agent_names[top_idx],
        mode=mode,
        baseline_signal=baseline,
        ablated_signal=ablated_signal,
        control_signal=control_signal,
        control_agent=agent_names[control_idx],
    )
