"""Kinematic Transfer Entropy (KTE): detect the *onset* of a collective
behavioral shift before it's fully established.

Raw transfer entropy tracks the *level* of information flow — it stays
elevated throughout an entire collusive regime, so a threshold on the level
alone fires late (only once the regime is well underway) and stays high long
after, which is uninformative about *when* the shift happened.

KTE instead tracks the *dynamics* of the TE time series: velocity,
acceleration, and jerk (1st/2nd/3rd derivatives), combined into one kinematic
norm. That norm spikes sharply right at the transition and returns to zero
once the new regime stabilizes — jerk in particular tends to rise before
velocity does, giving several extra timesteps of lead time versus watching
raw TE cross a threshold.

    KTE(t; tau) = || (TE'(t), tau * TE''(t), tau^2 * TE'''(t)) ||_2
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def kinematic_norm(ts: np.ndarray, tau: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """Kinematic norm of a 1-D time series: Gaussian-smoothed finite
    differences up to third order (jerk), combined via a weighted L2 norm.
    """
    s = gaussian_filter1d(ts, sigma) if len(ts) > 3 else ts.copy()
    v = np.gradient(s)
    a = np.gradient(v)
    j = np.gradient(a)
    sv, sa, sj = (gaussian_filter1d(x, sigma) if len(x) > 3 else x for x in (v, a, j))
    return np.sqrt(sv**2 + (tau * sa) ** 2 + (tau**2 * sj) ** 2)


def compute_kte(te_ts: np.ndarray, tau: float = 1.0, sigma: float = 1.5) -> dict:
    """Kinematic decomposition of a scalar TE time series.

    Returns a dict with the raw series, its smoothed velocity/acceleration/
    jerk, and the combined KTE norm — everything the detector and the plots
    need, computed once.
    """
    s = gaussian_filter1d(te_ts, sigma) if len(te_ts) > 3 else te_ts.copy()
    velocity = np.gradient(s)
    acceleration = np.gradient(velocity)
    jerk = np.gradient(acceleration)
    kte = kinematic_norm(te_ts, tau=tau, sigma=sigma)
    return {
        "te": te_ts,
        "te_smooth": s,
        "velocity": velocity,
        "acceleration": acceleration,
        "jerk": jerk,
        "kte": kte,
    }


def detect_transitions(kte: np.ndarray, burn_in: int = 10, n_std: float = 4.5, min_run: int = 4) -> list[int]:
    """Flag windows where KTE exceeds ``n_std`` robust standard deviations
    above its baseline *for at least ``min_run`` consecutive windows* — i.e.
    candidate onsets of a collective shift.

    Two things matter for keeping this from crying wolf on a quiet system
    (verified against ``make_null_scenario`` over dozens of seeds):

    - The baseline uses the median and MAD (scaled to be a consistent sigma
      estimator) of the *whole* series, not the mean/std of just the burn-in
      window. A true transition is a small fraction of the total windows, so
      it barely moves the median/MAD, but mean/std computed on a short
      burn-in slice is itself noisy and produces an unstable threshold from
      run to run.
    - ``min_run`` requires the excursion to persist: a genuine regime shift
      keeps KTE elevated while the new regime settles in, whereas per-window
      estimation noise on the TE series produces isolated 1-2 window blips
      even with zero real inter-agent coupling. Requiring a run filters
      those out without needing a much higher (and thus less sensitive)
      threshold.

    ``burn_in`` windows are never themselves flagged (there isn't enough
    history yet to trust a spike). Only the first window of each qualifying
    run is returned, since a single transition triggers many consecutive
    windows.
    """
    if len(kte) <= burn_in:
        return []
    med = np.median(kte)
    mad = np.median(np.abs(kte - med)) * 1.4826 + 1e-9
    threshold = med + n_std * mad

    above = kte[burn_in:] > threshold
    onsets = []
    run_start = None
    for offset, is_above in enumerate(above):
        idx = burn_in + offset
        if is_above:
            if run_start is None:
                run_start = idx
        else:
            if run_start is not None and idx - run_start >= min_run:
                onsets.append(run_start)
            run_start = None
    if run_start is not None and len(above) - (run_start - burn_in) >= min_run:
        onsets.append(run_start)
    return onsets


def emergence_score(kte: np.ndarray) -> np.ndarray:
    """KTE normalized to [0, 1] for reporting/plotting alongside other
    per-timestep scores that share that scale."""
    lo, hi = kte.min(), kte.max()
    if hi - lo < 1e-12:
        return np.zeros_like(kte)
    return (kte - lo) / (hi - lo)
