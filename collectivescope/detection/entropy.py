"""Shannon entropy of system states, for the collective-diversity signal that
complements transfer entropy (high entropy + low coordination = exploration;
low entropy + high coordination = a converged/collusive regime)."""

from __future__ import annotations

import numpy as np


def shannon_entropy(states: np.ndarray, bins: int = 10) -> float:
    """Histogram-based Shannon entropy (bits) of a (n_samples, d) array.

    Each dimension is discretized independently into ``bins`` equal-width
    bins, then joint entropy is computed over the resulting multi-index —
    fast and dependency-free, at the cost of some discretization bias for
    small samples.
    """
    states = np.atleast_2d(states)
    if states.ndim == 1:
        states = states[:, None]
    n_samples, d = states.shape
    if n_samples < 2:
        return 0.0

    digitized = np.zeros((n_samples, d), dtype=np.int64)
    for j in range(d):
        col = states[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            digitized[:, j] = 0
        else:
            edges = np.linspace(lo, hi, bins + 1)
            digitized[:, j] = np.clip(np.digitize(col, edges[1:-1]), 0, bins - 1)

    # collapse the per-dim bin indices into one joint symbol
    joint = np.zeros(n_samples, dtype=np.int64)
    for j in range(d):
        joint = joint * bins + digitized[:, j]

    _, counts = np.unique(joint, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def entropy_trajectory(states: np.ndarray, window: int = 20, step: int = 1, bins: int = 10) -> np.ndarray:
    """Sliding-window entropy of a collective. ``states`` is (T, n_agents, d);
    at each window we flatten the agent axis in, so entropy captures the
    *joint* diversity of the whole group's states, not one agent's.

    Returns an array of length ``ceil((T - window) / step) + 1``.
    """
    T = states.shape[0]
    flat = states.reshape(T, -1)
    out = []
    for t in range(window, T + 1, step):
        out.append(shannon_entropy(flat[t - window : t], bins=bins))
    return np.array(out)
