"""Self-contained synthetic scenario generator.

These scenarios need no external data or network access, so they power the
demo script, the dashboard's default view, and the test suite (where we need
known ground truth to check that detection/attribution/intervention actually
work). Each scenario mimics the shape of a real benchmark:

  - ``make_collusion_scenario``: like ColludeBench — a small driver subgroup
    starts secretly coordinating partway through an otherwise-independent run
    (a regime *shift*, with a known onset step and driver set).
  - ``make_coalition_scenario``: like Werewolf/Among Us — a coalition
    coordinates from the start (no shift to detect, but attribution ground
    truth is present throughout).
  - ``make_null_scenario``: no collusion at all, for false-positive checks.

All three return a ``Trajectory`` with ``states`` shape ``(T, n_agents, d)``.
"""

from __future__ import annotations

import numpy as np

from .base import Trajectory


def _agent_names(n_agents: int) -> list[str]:
    return [f"agent_{i}" for i in range(n_agents)]


def make_collusion_scenario(
    n_agents: int = 5,
    n_timesteps: int = 300,
    shift_step: int = 150,
    n_drivers: int = 2,
    d: int = 2,
    coupling: float = 0.85,
    noise: float = 0.3,
    seed: int = 0,
) -> Trajectory:
    """Independent agents for ``shift_step`` steps, then ``n_drivers`` of them
    start tightly coupling (one leads, the rest follow with lag-1 coupling +
    shared noise) while the remaining agents stay independent. This is the
    "regime shift with a known onset and a known driver subgroup" case.
    """
    rng = np.random.default_rng(seed)
    states = np.zeros((n_timesteps, n_agents, d))
    driver_idx = list(range(n_drivers))  # agent_0 is the leader, agent_1.. follow

    # independent phase: every agent is its own AR(1) process
    states[0] = rng.normal(size=(n_agents, d))
    for t in range(1, n_timesteps):
        if t < shift_step:
            for a in range(n_agents):
                states[t, a] = 0.9 * states[t - 1, a] + noise * rng.normal(size=d)
        else:
            leader = driver_idx[0]
            states[t, leader] = 0.9 * states[t - 1, leader] + noise * rng.normal(size=d)
            shared = rng.normal(size=d) * noise
            for a in range(n_agents):
                if a == leader:
                    continue
                if a in driver_idx:
                    # follower: driven by the leader's *previous* state (true TE)
                    states[t, a] = coupling * states[t - 1, leader] + (1 - coupling) * shared
                else:
                    states[t, a] = 0.9 * states[t - 1, a] + noise * rng.normal(size=d)

    return Trajectory(
        states=states,
        agent_names=_agent_names(n_agents),
        driver_agents={f"agent_{i}" for i in driver_idx},
        shift_step=shift_step,
        label="collusive",
        meta={"scenario": "collusion", "seed": seed, "coupling": coupling},
    )


def make_coalition_scenario(
    n_agents: int = 7,
    n_timesteps: int = 200,
    n_drivers: int = 2,
    d: int = 2,
    coupling: float = 0.75,
    noise: float = 0.35,
    seed: int = 0,
) -> Trajectory:
    """A driver coalition coordinates from t=0 (no onset to detect — this is
    the Werewolf/Among Us shape, where the pipeline's job is attribution, not
    localization)."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n_timesteps, n_agents, d))
    driver_idx = list(range(n_drivers))
    leader = driver_idx[0]

    states[0] = rng.normal(size=(n_agents, d))
    for t in range(1, n_timesteps):
        shared = rng.normal(size=d) * noise
        states[t, leader] = 0.9 * states[t - 1, leader] + noise * rng.normal(size=d)
        for a in range(n_agents):
            if a == leader:
                continue
            if a in driver_idx:
                states[t, a] = coupling * states[t - 1, leader] + (1 - coupling) * shared
            else:
                states[t, a] = 0.9 * states[t - 1, a] + noise * rng.normal(size=d)

    return Trajectory(
        states=states,
        agent_names=_agent_names(n_agents),
        driver_agents={f"agent_{i}" for i in driver_idx},
        shift_step=None,
        label="coalition",
        meta={"scenario": "coalition", "seed": seed, "coupling": coupling},
    )


def make_null_scenario(
    n_agents: int = 5,
    n_timesteps: int = 300,
    d: int = 2,
    noise: float = 0.3,
    seed: int = 0,
) -> Trajectory:
    """Every agent is an independent AR(1) process the whole time — no
    collective behavior, no drivers. Used to check the pipeline doesn't cry
    wolf on a quiet system."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n_timesteps, n_agents, d))
    states[0] = rng.normal(size=(n_agents, d))
    for t in range(1, n_timesteps):
        for a in range(n_agents):
            states[t, a] = 0.9 * states[t - 1, a] + noise * rng.normal(size=d)

    return Trajectory(
        states=states,
        agent_names=_agent_names(n_agents),
        driver_agents=set(),
        shift_step=None,
        label="honest",
        meta={"scenario": "null", "seed": seed},
    )
