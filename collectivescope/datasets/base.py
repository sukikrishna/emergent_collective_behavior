"""Common trajectory interface shared by every dataset adapter.

Every loader in this package — synthetic or real-benchmark — returns a
``Trajectory``. The detection/attribution/intervention modules only ever see
this type, so a new benchmark can be plugged in by writing one function that
returns one of these; nothing downstream needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trajectory:
    """A multi-agent trajectory plus whatever ground truth is available.

    states : (T, n_agents, d) float array. Per-agent state at each timestep —
        e.g. a bid, an embedded message, a contribution fraction.
    agent_names : names for the n_agents axis, in index order.
    codes : optional (n_agents, T) int array of discrete action/tool-call
        codes, for the symbolic transfer-entropy front-end. Not every dataset
        has a natural discretization, so this is optional.
    driver_agents : ground-truth set of agent names that *drive* the
        collective behavior (colluders, werewolves, impostors, ...). None if
        the dataset has no attribution ground truth (e.g. it's a pure
        detection benchmark).
    shift_step : ground-truth timestep at which a collective regime shift
        occurs (e.g. collusion begins, cooperation collapses). None if the
        behavior is present for the whole run or there's no known onset.
    label : free-form regime label (e.g. "collusive", "honest", "collapsed"),
        for run-level detection benchmarks that classify whole runs rather
        than localizing an onset.
    meta : anything else worth carrying along (source dataset name, run id,
        scenario config) for reporting/plots.
    """

    states: np.ndarray
    agent_names: list[str]
    codes: Optional[np.ndarray] = None
    driver_agents: Optional[set[str]] = None
    shift_step: Optional[int] = None
    label: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.states.ndim != 3:
            raise ValueError(f"states must be (T, n_agents, d), got shape {self.states.shape}")
        T, n_ag, d = self.states.shape
        if len(self.agent_names) != n_ag:
            raise ValueError(
                f"agent_names has {len(self.agent_names)} entries but states has "
                f"{n_ag} agents"
            )
        if self.codes is not None and self.codes.shape != (n_ag, T):
            raise ValueError(
                f"codes must be (n_agents, T) = {(n_ag, T)}, got {self.codes.shape}"
            )
        if self.driver_agents is not None:
            unknown = self.driver_agents - set(self.agent_names)
            if unknown:
                raise ValueError(f"driver_agents contains unknown agent names: {unknown}")

    @property
    def n_timesteps(self) -> int:
        return self.states.shape[0]

    @property
    def n_agents(self) -> int:
        return self.states.shape[1]

    def driver_mask(self) -> Optional[np.ndarray]:
        """Boolean (n_agents,) array, True where the agent is a ground-truth
        driver. None if this trajectory has no attribution ground truth."""
        if self.driver_agents is None:
            return None
        return np.array([name in self.driver_agents for name in self.agent_names])

    def agent_index(self, name: str) -> int:
        return self.agent_names.index(name)
