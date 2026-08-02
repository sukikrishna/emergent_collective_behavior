"""Wires the three stages together against a single ``Trajectory``: detect,
attribute, (optionally) validate via intervention. This is the one function
the demo script, dashboard, and tests all call, so the window/step/threshold
defaults live in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attribution.influence_graph import InfluenceGraph, kca_attribution
from .datasets.base import Trajectory
from .detection.kte import compute_kte, detect_transitions
from .detection.transfer_entropy import mean_te_series, te_matrix_over_time
from .intervention.ablation import InterventionResult, validate_intervention


@dataclass
class PipelineResult:
    trajectory: Trajectory
    window: int
    step: int
    te_mats: np.ndarray          # (n_agents, n_agents, n_windows)
    te_series: np.ndarray        # (n_windows,) scalar mean TE
    kte: dict                    # compute_kte() output
    transitions: list[int]       # detected onset window-indices
    influence_graph: InfluenceGraph
    attribution: dict[str, np.ndarray]  # method name -> per-agent score
    intervention: InterventionResult | None = None

    def onset_timesteps(self) -> list[int]:
        """Detected onsets converted from window-index back to the original
        trajectory's timestep axis, for comparing against ground truth."""
        return [self.window + idx * self.step for idx in self.transitions]


def _default_window(n_timesteps: int) -> int:
    return int(np.clip(n_timesteps // 6, 10, 50))


def run_pipeline(
    traj: Trajectory,
    window: int | None = None,
    step: int | None = None,
    conditional: bool = True,
    n_std: float = 4.5,
    run_intervention: bool = True,
    intervention_mode: str = "silence",
    attribution_method: str = "kca",
) -> PipelineResult:
    """Run detection + attribution (+ intervention) end to end on one
    trajectory, with sensible defaults for the sliding-window size and
    detection threshold. See ``examples/run_end_to_end_demo.py`` for a worked
    example and ``docs/design.md`` for how each stage maps to the proposal.
    """
    T = traj.n_timesteps
    window = window or _default_window(T)
    step = step or max(1, window // 12)

    te_mats = te_matrix_over_time(traj.states, window=window, conditional=conditional, step=step)
    te_series = mean_te_series(te_mats)
    kte = compute_kte(te_series, sigma=max(1.5, window / 20))

    burn_in = max(5, int(0.3 * len(kte["kte"])))
    transitions = detect_transitions(kte["kte"], burn_in=burn_in, n_std=n_std)

    ig = InfluenceGraph(te_mats, agent_names=traj.agent_names)
    attribution = ig.summary()
    if attribution_method == "kca":
        attribution["kca"] = kca_attribution(traj.states, window=window, conditional=conditional)

    intervention = None
    if run_intervention:
        primary_scores = attribution.get(attribution_method, attribution["pagerank"])
        intervention = validate_intervention(
            traj.states, traj.agent_names, primary_scores, mode=intervention_mode, conditional=conditional
        )

    return PipelineResult(
        trajectory=traj,
        window=window,
        step=step,
        te_mats=te_mats,
        te_series=te_series,
        kte=kte,
        transitions=transitions,
        influence_graph=ig,
        attribution=attribution,
        intervention=intervention,
    )
