"""Attribution: turn a directed influence signal into a map of who initiates,
propagates, or amplifies collective behavior.

Rather than averaging the pairwise transfer-entropy matrix down to a single
system-wide score (as the detection stage does, for KTE), this module keeps
the full N x N directed graph and asks per-agent questions of it:

  - out-strength(i): how much i drives everyone else — a large value means i
    is a source of coordination.
  - in-strength(j): how much j is driven — a large value means j is a
    follower.
  - hub-score(k) = out-strength(k) * in-strength(k): agents that both receive
    and relay a lot of information — potential bottlenecks/relays.
  - causal-dominance(i) = max_j W[i,j] / mean_j W[:,j]: agents with one
    disproportionately strong outgoing edge, vs. broadly-average influence.
  - PageRank on the row-normalized W: recursive influence — an agent that
    drives *other high-influence* agents scores higher than one that only
    drives peripheral agents, which plain out-strength can't distinguish.

KCA (Kinematic Causal Attribution) composes the two pipeline stages: apply the
detection stage's kinematic operator to every directed, time-resolved
conditional-TE pair (rather than to the scalar mean series), then sum each
agent's outgoing kinematic flow. This scores *which* agent's causal out-flow
is changing at a transition, not just which agent's average flow is largest —
useful when the driver only becomes dominant partway through a run.
"""

from __future__ import annotations

import numpy as np

from ..detection.kte import kinematic_norm
from ..detection.transfer_entropy import te_matrix_granger, te_matrix_over_time


class InfluenceGraph:
    """Wraps a directed TE matrix (static, or a time series of them) and
    exposes the per-agent attribution metrics described above."""

    def __init__(self, W: np.ndarray, agent_names: list[str] | None = None):
        """W is either (n_agents, n_agents) — one static graph — or
        (n_agents, n_agents, T) — a time series of graphs, in which case every
        metric below is computed per-timestep and returned as (n_agents, T)."""
        if W.ndim == 2:
            W = W[:, :, None]
        self.W = W
        self.n_agents = W.shape[0]
        self.agent_names = agent_names or [f"agent_{i}" for i in range(self.n_agents)]

    def out_strength(self) -> np.ndarray:
        """(n_agents, T): total outgoing influence per agent per timestep."""
        return self.W.sum(axis=1)

    def in_strength(self) -> np.ndarray:
        """(n_agents, T): total incoming influence per agent per timestep."""
        return self.W.sum(axis=0)

    def hub_score(self) -> np.ndarray:
        return self.out_strength() * self.in_strength()

    def causal_dominance(self) -> np.ndarray:
        out = self.out_strength()
        row_max = self.W.max(axis=1)
        mean_out = out.mean(axis=0, keepdims=True) / self.n_agents + 1e-12
        return row_max / mean_out

    def pagerank(self, damping: float = 0.85, n_iter: int = 100) -> np.ndarray:
        """PageRank of the row-normalized influence graph, per timestep.

        Row-normalizing turns W into a transition matrix over "who influences
        whom next"; the stationary distribution of that walk is the recursive
        influence score. Isolated agents (no outgoing edges) are treated as
        transitioning uniformly to everyone, the standard PageRank fix for
        dangling nodes.
        """
        n, T = self.n_agents, self.W.shape[-1]
        scores = np.zeros((n, T))
        for t in range(T):
            Wt = self.W[:, :, t]
            row_sums = Wt.sum(axis=1, keepdims=True)
            dangling = (row_sums.squeeze(-1) < 1e-12)
            P = np.divide(Wt, row_sums, out=np.zeros_like(Wt), where=row_sums > 1e-12)
            P[dangling] = 1.0 / n

            r = np.full(n, 1.0 / n)
            for _ in range(n_iter):
                r = (1 - damping) / n + damping * (P.T @ r)
            scores[:, t] = r
        return scores

    def summary(self) -> dict[str, np.ndarray]:
        """All metrics, time-averaged to one score per agent — the ranking
        used for driver-vs-non-driver evaluation and dashboard bar charts."""
        return {
            "out_strength": self.out_strength().mean(axis=1),
            "in_strength": self.in_strength().mean(axis=1),
            "hub_score": self.hub_score().mean(axis=1),
            "causal_dominance": self.causal_dominance().mean(axis=1),
            "pagerank": self.pagerank().mean(axis=1),
        }


def build_influence_graph(
    traj: np.ndarray, agent_names: list[str] | None = None, window: int | None = None, conditional: bool = True
) -> InfluenceGraph:
    """Convenience constructor: run conditional Granger TE over ``traj``
    (T, n_agents, d) and wrap the result. If ``window`` is given, produces a
    time-resolved graph via a sliding window; otherwise one static graph over
    the whole trajectory."""
    if window is None:
        W = te_matrix_granger(traj, conditional=conditional)
    else:
        W = te_matrix_over_time(traj, window=window, conditional=conditional)
    return InfluenceGraph(W, agent_names=agent_names)


def kca_matrix(traj: np.ndarray, window: int | None = None, conditional: bool = True, tau: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair kinematic norm of the time-resolved conditional-TE series.

    Returns (kmat, te_mats): kmat is (n_agents, n_agents, T), the kinematic
    norm applied independently to every directed pair's TE(t); te_mats is the
    underlying (n_agents, n_agents, T) TE series it was computed from.
    """
    T_total, n_agents, _ = traj.shape
    if window is None:
        # Match pipeline.py's _default_window: a short window here makes the
        # per-window TE estimate noisy, and the kinematic derivative
        # operators amplify that noise, favoring whichever agent's
        # trajectory happens to be noisiest rather than the real driver.
        window = int(np.clip(T_total // 6, 10, 50))
    te_mats = te_matrix_over_time(traj, window=window, conditional=conditional)
    kmat = np.zeros_like(te_mats)
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            kmat[i, j] = kinematic_norm(te_mats[i, j], tau=tau)
    return kmat, te_mats


def kca_attribution(traj: np.ndarray, window: int | None = None, conditional: bool = True, agg: str = "sum") -> np.ndarray:
    """Per-agent KCA driver score (higher = more likely to be driving the
    change in collective behavior).

    agg: "sum" totals each agent's kinematic out-flow over the whole run;
    "onset" scores each agent only at the timestep where the *mean* pairwise
    kinematic signal peaks (i.e. attribution right at the detected
    transition, rather than averaged across the whole run).
    """
    kmat, _ = kca_matrix(traj, window=window, conditional=conditional)
    out_flow = kmat.sum(axis=1)  # (n_agents, T)
    if agg == "sum":
        return out_flow.sum(axis=1)
    if agg == "onset":
        n_agents = kmat.shape[0]
        mask = ~np.eye(n_agents, dtype=bool)
        mean_k = kmat.transpose(2, 0, 1)[:, mask].mean(axis=1)
        t_star = int(np.argmax(mean_k)) if mean_k.size else 0
        return out_flow[:, t_star]
    raise ValueError(f"unknown agg {agg!r}")
