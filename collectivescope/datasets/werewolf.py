"""Werewolf adapter — LLM social-deduction / hidden-role deception benchmark.

Every player is an LLM agent (unlike Among Us, which mixes human and LLM
players), so the driver-attribution question is clean: drivers = the
werewolf coalition. Private night chat is excluded — it would leak the
coalition directly — so the trajectory is built from public speeches and vote
justifications only.

This adapter expects Werewolf game logs checked out under
``$DATA_ROOT/werewolf/`` — see ``data/README.md``. Turning each agent's
message text into a numeric state (message embedding -> shared low-dim
projection) requires an embedding backend; this module uses a dependency-free
hashing embedder by default so it runs with no extra installs, and accepts a
custom ``embed_fn`` for a real sentence-embedding model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .base import Trajectory


def _data_root() -> Path:
    root = os.environ.get("DATA_ROOT", "./data/raw")
    return Path(root) / "werewolf"


def is_available() -> bool:
    root = _data_root()
    return root.exists() and any(root.rglob("*.json"))


def load_games(max_games: int | None = None) -> list[dict]:
    root = _data_root()
    if not is_available():
        raise FileNotFoundError(
            f"No Werewolf game logs found under {root}. "
            "See data/README.md for how to obtain the dataset and set DATA_ROOT."
        )
    paths = sorted(root.rglob("game_*.json"))
    if max_games:
        paths = paths[:max_games]
    games = []
    for p in paths:
        with open(p) as f:
            games.append(json.load(f))
    return games


def _hash_embed(text: str, dim: int = 16) -> np.ndarray:
    """Dependency-free fallback embedder: hashes tokens into a fixed-size bag,
    so the pipeline runs without sentence-transformers installed. Swap in a
    real embedder via ``embed_fn`` for anything beyond a smoke test."""
    vec = np.zeros(dim)
    for tok in text.lower().split():
        vec[hash(tok) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_trajectory(game: dict, embed_fn=None, out_dim: int = 2) -> Trajectory:
    """Convert one Werewolf game dict into a ``Trajectory``.

    Expects the game's own schema: a chronological list of public turns, each
    with a ``player`` and ``text``, plus a ``werewolves`` ground-truth field.
    Message embeddings are pooled across all agents/turns and projected onto
    their top ``out_dim`` principal components so every agent's coordinates
    live in one shared, comparable space (a prerequisite for transfer
    entropy).
    """
    embed_fn = embed_fn or _hash_embed
    agent_names = sorted(game["players"])
    turns = game["turns"]  # [{player, text, day}, ...], public speech only

    raw_dim = len(embed_fn(turns[0]["text"])) if turns else out_dim
    all_vecs = np.array([embed_fn(t["text"]) for t in turns]) if turns else np.zeros((0, raw_dim))

    if len(all_vecs) >= out_dim:
        centered = all_vecs - all_vecs.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:out_dim].T
        proj = centered @ basis
    else:
        proj = np.zeros((len(all_vecs), out_dim))

    T = max((t.get("day", 0) for t in turns), default=0) + 1
    states = np.zeros((T, len(agent_names), out_dim))
    last_seen = np.zeros((len(agent_names), out_dim))
    for k, t in enumerate(turns):
        a = agent_names.index(t["player"])
        last_seen[a] = proj[k]
        day = t.get("day", 0)
        states[day, a] = proj[k]
    # zero-order hold: carry the last known state forward for agents silent that turn
    for t in range(1, T):
        for a in range(len(agent_names)):
            if np.allclose(states[t, a], 0):
                states[t, a] = states[t - 1, a]

    werewolves = set(game.get("werewolves", []))
    return Trajectory(
        states=states,
        agent_names=agent_names,
        driver_agents=werewolves,
        shift_step=None,
        label="deceptive" if werewolves else None,
        meta={"source": "werewolf", "game_id": game.get("id")},
    )
