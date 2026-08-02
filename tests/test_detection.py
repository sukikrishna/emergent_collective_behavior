"""Detection stage: does KTE find the true onset promptly, and does it stay
quiet on a trajectory with no collective behavior at all?

The false-positive check (test_null_scenario_has_low_false_positive_rate) is
a regression guard for a real bug found during development: the original
single-window, burn-in-mean/std threshold in detect_transitions() fired on
roughly 13/20 seeds of make_null_scenario even though those trajectories have
zero real inter-agent coupling. The fix (median/MAD baseline over the whole
series + a minimum consecutive-window run length) drove that to 0/30 in
manual sweeps; this test keeps it there.
"""

from __future__ import annotations

from collectivescope.datasets import make_coalition_scenario, make_collusion_scenario, make_null_scenario
from collectivescope.pipeline import run_pipeline


def test_collusion_onset_detected_promptly():
    traj = make_collusion_scenario(
        n_agents=5, n_timesteps=300, shift_step=150, n_drivers=2, coupling=0.9, noise=0.2, seed=1
    )
    result = run_pipeline(traj, run_intervention=False)
    onsets = result.onset_timesteps()
    assert onsets, "expected at least one detected transition"
    # detection lags the true shift by one sliding-window's worth of evidence;
    # it must not be missed, and shouldn't lag by more than ~2 windows.
    assert 0 <= onsets[0] - traj.shift_step <= 2 * result.window


def test_coalition_scenario_has_no_onset_to_find():
    # coalition drivers coordinate from t=0 -- there is no regime *shift*,
    # so detection (which looks for a change) should not report one.
    traj = make_coalition_scenario(n_agents=7, n_timesteps=200, n_drivers=2, coupling=0.8, seed=1)
    result = run_pipeline(traj, run_intervention=False)
    assert result.onset_timesteps() == []


def test_null_scenario_has_low_false_positive_rate():
    n_seeds = 30
    false_positives = 0
    for seed in range(n_seeds):
        traj = make_null_scenario(n_agents=5, n_timesteps=300, seed=seed)
        result = run_pipeline(traj, run_intervention=False)
        if result.transitions:
            false_positives += 1
    assert false_positives / n_seeds <= 0.10, (
        f"false positive rate too high: {false_positives}/{n_seeds}"
    )
