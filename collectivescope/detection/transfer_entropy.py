"""Transfer entropy: directed information flow between agents.

Two estimators, matching two kinds of agent state:

  - Granger TE (continuous states, e.g. embedded messages, numeric actions):
    i -> j is the log variance-ratio between an AR(1) model of j that includes
    i's lagged state and one that doesn't. Conditioning on the rest of the
    group (the other agents' lagged states) turns this into *conditional* TE,
    which removes common-driver/relay confounds — the difference between "i
    actually influences j" and "i and j are both driven by a third agent k".

  - Symbolic TE (discrete action/tool-call codes): Schreiber's count-based
    estimator, i -> j = I(j_future ; i_past | j_past), via joint frequency
    tables over a window.

Both return a directed (n_agents, n_agents) matrix per window, with the
diagonal left at 0 (no self-influence). This module intentionally implements
the estimators fresh rather than importing them — the versions in
~/Documents/KTE (causal_te_attribution.py, attribution_study/method/
conditional_te.py) were the reference for the math, but that repo is a
separate, unrelated project.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.ndimage import uniform_filter1d


def _conditional_granger(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None, lag: int = 1,
                          ridge: float = 1e-2) -> float:
    """Conditional Granger TE x -> y given conditioning set z, shape (T, d_z).

    Returns log(var_restricted / var_full), clamped to >= 0 (a negative value
    means the full model didn't help predict y, i.e. no detectable flow).

    Fit with ridge-regularized least squares rather than plain ``lstsq``: the
    full model always has more regressors than the restricted one (it adds
    x_past on top of everything z has), so on a short window an unregularized
    fit spuriously lowers var_full just by having more free parameters to
    chase noise with — inflating TE for every pair, not only real senders.
    Ridge shrinkage keeps that degrees-of-freedom gap from dominating the
    variance-ratio estimate, which matters most exactly when the conditioning
    set z is large relative to the window (i.e. conditional TE with many
    agents).
    """
    T = len(y)
    if T < lag + 4:
        return 0.0

    y_future = y[lag:]
    y_past = y[:-lag]
    x_past = x[:-lag]

    def _design(*cols: np.ndarray) -> np.ndarray:
        return np.column_stack([c.reshape(len(c), -1) for c in cols] + [np.ones(len(y_future))])

    if z is not None:
        z_past = z[:-lag]
        X_restricted = _design(y_past, z_past)
        X_full = _design(y_past, x_past, z_past)
    else:
        X_restricted = _design(y_past)
        X_full = _design(y_past, x_past)

    target = y_future.reshape(len(y_future), -1)

    def _residual_var(X: np.ndarray) -> float:
        n_features = X.shape[1]
        A = X.T @ X + ridge * np.eye(n_features)
        b = X.T @ target
        beta = np.linalg.solve(A, b)
        resid = target - X @ beta
        return float(np.var(resid) + 1e-12)

    var_r = _residual_var(X_restricted)
    var_f = _residual_var(X_full)
    te = np.log(var_r / var_f)
    return max(te, 0.0)


def pairwise_te_granger(traj: np.ndarray, i: int, j: int, conditional: bool = True, lag: int = 1) -> float:
    """Granger TE for a single directed pair i -> j over the whole ``traj``
    (T, n_agents, d), conditioning on every other agent if ``conditional``."""
    n_agents = traj.shape[1]
    x, y = traj[:, i], traj[:, j]
    if conditional:
        others = [k for k in range(n_agents) if k not in (i, j)]
        z = traj[:, others] if others else None
        if z is not None:
            z = z.reshape(z.shape[0], -1)
    else:
        z = None
    return _conditional_granger(x, y, z, lag=lag)


def te_matrix_granger(traj: np.ndarray, conditional: bool = True, lag: int = 1) -> np.ndarray:
    """Full directed (n_agents, n_agents) conditional-TE matrix for one window
    of ``traj`` (T, n_agents, d). W[i, j] = flow i -> j."""
    n_agents = traj.shape[1]
    W = np.zeros((n_agents, n_agents))
    for i, j in product(range(n_agents), range(n_agents)):
        if i == j:
            continue
        W[i, j] = pairwise_te_granger(traj, i, j, conditional=conditional, lag=lag)
    return W


def te_matrix_over_time(traj: np.ndarray, window: int, conditional: bool = True, step: int = 1) -> np.ndarray:
    """Sliding-window conditional-TE matrix time series.

    Returns shape (n_agents, n_agents, n_windows).
    """
    T, n_agents, _ = traj.shape
    mats = []
    for t in range(window, T + 1, step):
        mats.append(te_matrix_granger(traj[t - window : t], conditional=conditional))
    if not mats:
        return np.zeros((n_agents, n_agents, 0))
    return np.stack(mats, axis=-1)


def pairwise_te_symbolic(src: np.ndarray, dst: np.ndarray) -> float:
    """Count-based (Schreiber) TE src -> dst for one window of discrete codes.

    Uses history length 1 for both series, matching the AR(1) order of the
    continuous estimator above. ``src``/``dst`` are 1-D integer arrays of
    equal length (one window's worth of symbols).
    """
    if len(dst) < 4:
        return 0.0
    jf, jp, ip = dst[1:], dst[:-1], src[:-1]
    n = len(jf)

    def _joint_counts(*cols: np.ndarray) -> dict:
        counts: dict = {}
        for row in zip(*cols):
            counts[row] = counts.get(row, 0) + 1
        return counts

    c_jf_jp_ip = _joint_counts(jf, jp, ip)
    c_jp_ip = _joint_counts(jp, ip)
    c_jf_jp = _joint_counts(jf, jp)
    c_jp = _joint_counts(jp)

    te = 0.0
    for (f, p, i), cnt in c_jf_jp_ip.items():
        p_fpi = cnt / n
        p_f_given_pi = cnt / c_jp_ip[(p, i)]
        p_f_given_p = c_jf_jp[(f, p)] / c_jp[p]
        if p_f_given_pi > 0 and p_f_given_p > 0:
            te += p_fpi * np.log2(p_f_given_pi / p_f_given_p)
    return max(te, 0.0)


def te_matrix_symbolic(codes: np.ndarray) -> np.ndarray:
    """Full directed (n_agents, n_agents) symbolic-TE matrix for one window of
    ``codes`` (n_agents, T) discrete action codes."""
    n_agents = codes.shape[0]
    W = np.zeros((n_agents, n_agents))
    for i, j in product(range(n_agents), range(n_agents)):
        if i == j:
            continue
        W[i, j] = pairwise_te_symbolic(codes[i], codes[j])
    return W


def mean_te_series(mats: np.ndarray) -> np.ndarray:
    """Collapse a (n_agents, n_agents, n_windows) TE matrix series to a scalar
    TE(t) by averaging over all directed off-diagonal pairs — the input to
    the KTE kinematic operator, which only makes sense on a scalar series."""
    n_agents = mats.shape[0]
    off_diag = ~np.eye(n_agents, dtype=bool)
    return np.array([mats[:, :, t][off_diag].mean() for t in range(mats.shape[-1])])


def smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Light uniform smoothing shared by the detection/attribution modules so
    plotted series aren't dominated by per-window estimation noise."""
    return uniform_filter1d(x, size=window, mode="nearest")
