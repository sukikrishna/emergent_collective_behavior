"""CooperBench adapter — LLM-agent collaborative coding benchmark.

CooperBench runs two coding agents (each assigned a feature) through a
shared repo, optionally cooperating on merge/coordination. Unlike the other
three adapters, this benchmark's collective-behavior question is not
collusion/deception but *coordination quality*: does the agents'
communication pattern predict whether both features land cleanly?

Expects CooperBench run-log directories checked out under
``$DATA_ROOT/cooperbench/logs/<run_name>/coop/<repo>/<task_id>/f<a>_f<b>/`` —
see ``data/README.md``. This mirrors CooperBench's own on-disk layout, so a
run of ``cooperbench run --setting coop`` can be pointed at directly via
DATA_ROOT with no reshaping.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .base import Trajectory


def _data_root() -> Path:
    root = os.environ.get("DATA_ROOT", "./data/raw")
    return Path(root) / "cooperbench" / "logs"


def is_available() -> bool:
    root = _data_root()
    return root.exists() and any(root.rglob("result.json"))


def load_run_dirs(run_name: str | None = None) -> list[Path]:
    root = _data_root()
    if not is_available():
        raise FileNotFoundError(
            f"No CooperBench run logs found under {root}. "
            "See data/README.md for how to obtain/produce the dataset and set DATA_ROOT."
        )
    base = root / run_name if run_name else root
    return sorted({p.parent for p in base.rglob("result.json")})


def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_trajectory(run_dir: Path, n_steps_max: int = 200) -> Trajectory:
    """Convert one CooperBench ``f<a>_f<b>`` run directory into a
    ``Trajectory``. States are two channels per agent: a step-progress signal
    (assistant turns taken so far, normalized) and a communication-burst
    signal (whether the agent sent a cross-agent message that step) — a
    coarse but dependency-free proxy for "is this agent acting" / "is this
    agent coordinating".
    """
    result = _read_json(run_dir / "result.json") or {}
    agents_meta = result.get("agents", {})
    agent_names = sorted(agents_meta.keys())

    traj_by_agent = {}
    for fid in agent_names:
        traj = _read_json(run_dir / f"agent{fid}_traj.json") or {}
        traj_by_agent[fid] = traj.get("messages", [])

    conv = _read_json(run_dir / "conversation.json") or []

    T = min(max((len(m) for m in traj_by_agent.values()), default=1), n_steps_max)
    states = np.zeros((T, len(agent_names), 2))
    for a, fid in enumerate(agent_names):
        n_msgs = len(traj_by_agent[fid])
        for t in range(T):
            states[t, a, 0] = min(t, n_msgs) / max(n_msgs, 1)  # progress
    for msg in conv:
        t = min(int(msg.get("timestamp", 0)) % T, T - 1)
        sender = str(msg.get("from"))
        if sender in agent_names:
            states[t, agent_names.index(sender), 1] = 1.0

    eval_data = _read_json(run_dir / "eval.json") or {}
    both_passed = eval_data.get("both_passed")

    return Trajectory(
        states=states,
        agent_names=agent_names,
        driver_agents=None,  # no attribution ground truth; outcome is run-level
        shift_step=None,
        label="both_passed" if both_passed else "failed",
        meta={"source": "cooperbench", "run_dir": str(run_dir)},
    )
