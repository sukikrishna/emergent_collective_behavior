"""ColludeBench adapter — LLM price-collusion benchmark.

Source: "Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent
LLMs" (ColludeBench-v0). Scenario used: ``first_price_auction`` (3 bidders, 10
rounds); colluders are the ground-truth driver set, honest runs have none.

This adapter expects the ColludeBench repo (or just its run-log JSON files)
checked out under ``$DATA_ROOT/colludebench/`` — see ``data/README.md`` for
where to get it. It is not bundled with this repo.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

from .base import Trajectory


def _data_root() -> Path:
    root = os.environ.get("DATA_ROOT", "./data/raw")
    return Path(root) / "colludebench"


def is_available() -> bool:
    """Whether real ColludeBench run logs are present at DATA_ROOT."""
    root = _data_root()
    return root.exists() and any(root.rglob("*.json"))


def load_runs(scenario: str = "first_price_auction", max_runs: int | None = None) -> list[dict]:
    """Load raw run dicts from ``$DATA_ROOT/colludebench``. Raises with a
    clear message (rather than silently returning []) if the data isn't
    there, since a silently-empty benchmark is easy to mistake for "ran and
    found nothing"."""
    root = _data_root()
    if not is_available():
        raise FileNotFoundError(
            f"No ColludeBench run logs found under {root}. "
            "See data/README.md for how to obtain the dataset and set DATA_ROOT."
        )
    paths = sorted(glob.glob(str(root / "**" / f"*{scenario}*.json"), recursive=True))
    if max_runs:
        paths = paths[:max_runs]
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def build_trajectory(run: dict) -> Trajectory:
    """Convert one ColludeBench run dict into a ``Trajectory``.

    Expects the run's own schema: a per-round, per-agent action/reward
    record and a ``colluders`` ground-truth field, matching the structure
    ColludeBench run logs are saved in.
    """
    agent_names = sorted(run["agents"])
    rounds = run["rounds"]
    T = len(rounds)
    states = np.zeros((T, len(agent_names), 2))
    for t, rd in enumerate(rounds):
        for a, name in enumerate(agent_names):
            rec = rd.get(name, {})
            states[t, a, 0] = float(rec.get("action", np.nan))
            states[t, a, 1] = float(rec.get("reward", np.nan))
    states = np.nan_to_num(states, nan=0.0)

    colluders = set(run.get("colluders", []))
    return Trajectory(
        states=states,
        agent_names=agent_names,
        driver_agents=colluders,
        shift_step=None,  # collusion is on for the whole run in this scenario
        label="collusive" if colluders else "honest",
        meta={"scenario": run.get("scenario", "first_price_auction"), "source": "colludebench"},
    )
