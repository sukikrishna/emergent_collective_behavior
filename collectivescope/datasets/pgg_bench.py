"""PGG-Bench adapter — Public Goods Game ("Contribute & Punish") benchmark.

A cooperation genre, complementing the collusion (ColludeBench) and deception
(Werewolf) datasets: the collective shift of interest is cooperation ->
free-riding collapse, so this is primarily a DETECTION benchmark (does the
pipeline notice the collapse?) rather than an attribution one, though the
lowest-mean-contributor makes a reasonable free-rider driver label.

5 LLM players, 10 rounds; each round every player contributes some fraction of
its tokens to a shared pool (multiplied and split evenly). The natural
per-agent signal is the contribution fraction, bounded in [0, 1].

Expects PGG-Bench logs (jsonl, one ``contribution`` event per player per
round) checked out under ``$DATA_ROOT/pgg_bench/`` — see ``data/README.md``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from .base import Trajectory

COLLAPSE_DROP = 0.20  # early-vs-late mean contribution-fraction drop -> "collapsed"


def _data_root() -> Path:
    root = os.environ.get("DATA_ROOT", "./data/raw")
    return Path(root) / "pgg_bench"


def is_available() -> bool:
    root = _data_root()
    return root.exists() and any(root.rglob("game_*.jsonl"))


def _pidx(player_id: str) -> int | None:
    m = re.match(r"Player(\d+)", player_id)
    return int(m.group(1)) - 1 if m else None


def load_games(max_games: int | None = None) -> list[dict]:
    root = _data_root()
    if not is_available():
        raise FileNotFoundError(
            f"No PGG-Bench logs found under {root}. "
            "See data/README.md for how to obtain the dataset and set DATA_ROOT."
        )
    paths = sorted(root.rglob("game_*.jsonl"))
    if max_games:
        paths = paths[:max_games]
    games = []
    for p in paths:
        events = [json.loads(line) for line in open(p) if line.strip()]
        games.append({"path": str(p), "events": events})
    return games


def build_trajectory(game: dict) -> Trajectory:
    """Convert one PGG-Bench game dict into a ``Trajectory`` of per-round
    contribution fractions, shape (T=rounds, n_agents, 1)."""
    events = game["events"]
    contributions = [e for e in events if e.get("type") == "contribution"]
    player_ids = sorted({e["player_id"] for e in contributions}, key=lambda p: _pidx(p) or 0)
    n_agents = len(player_ids)
    n_rounds = max((e["round"] for e in contributions), default=-1) + 1

    states = np.zeros((n_rounds, n_agents, 1))
    for e in contributions:
        a = player_ids.index(e["player_id"])
        r = e["round"]
        tokens = e.get("current_tokens", 0)
        frac = e["contribution"] / tokens if tokens else 0.0
        states[r, a, 0] = np.clip(frac, 0.0, 1.0)

    mean_by_round = states[:, :, 0].mean(axis=1)
    early = mean_by_round[:2].mean() if n_rounds >= 2 else mean_by_round.mean()
    late = mean_by_round[-2:].mean() if n_rounds >= 2 else mean_by_round.mean()
    collapsed = (early - late) >= COLLAPSE_DROP

    free_rider_idx = int(states[:, :, 0].mean(axis=0).argmin()) if n_agents else None
    driver = {player_ids[free_rider_idx]} if free_rider_idx is not None else set()

    return Trajectory(
        states=states,
        agent_names=list(player_ids),
        driver_agents=driver,
        shift_step=None,
        label="collapsed" if collapsed else "sustained",
        meta={"source": "pgg_bench", "path": game.get("path")},
    )
