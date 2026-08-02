#!/usr/bin/env python3
"""End-to-end CollectiveScope demo on a synthetic collusion scenario.

Runs all three pipeline stages against a scenario with known ground truth
(a leader + follower subgroup that starts secretly coordinating partway
through the run) and prints a report: did detection find the onset, did
attribution rank the true drivers above the honest agents, and did the
intervention's causal prediction hold?

Usage:
    python examples/run_end_to_end_demo.py
    python examples/run_end_to_end_demo.py --scenario coalition --out-dir outputs
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from collectivescope import run_pipeline
from collectivescope.datasets import make_coalition_scenario, make_collusion_scenario, make_null_scenario
from collectivescope.viz.plots import plot_attribution_bars, plot_influence_graph, plot_kte_timeline

SCENARIOS = {
    "collusion": lambda: make_collusion_scenario(
        n_agents=5, n_timesteps=300, shift_step=150, n_drivers=2, coupling=0.9, noise=0.2, seed=1
    ),
    "coalition": lambda: make_coalition_scenario(n_agents=7, n_timesteps=200, n_drivers=2, coupling=0.8, seed=1),
    "null": lambda: make_null_scenario(n_agents=5, n_timesteps=300, seed=1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="collusion")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--attribution-method", default="kca")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    traj = SCENARIOS[args.scenario]()
    print(f"=== CollectiveScope end-to-end demo — scenario: {args.scenario!r} ===")
    print(f"agents: {traj.agent_names}")
    print(f"ground-truth drivers: {traj.driver_agents or '(none)'}")
    print(f"ground-truth shift step: {traj.shift_step}")
    print()

    result = run_pipeline(traj, attribution_method=args.attribution_method)

    print("--- Stage 1: Detection ---")
    onsets = result.onset_timesteps()
    print(f"window={result.window}, step={result.step}")
    print(f"detected onset timesteps: {onsets}")
    if traj.shift_step is not None:
        if onsets:
            lead = traj.shift_step - onsets[0]
            print(f"first detection vs. true shift ({traj.shift_step}): delta={-lead:+d} steps "
                  f"({'early/at onset' if lead >= 0 else 'late'})")
        else:
            print("no transition detected (missed).")
    print()

    print("--- Stage 2: Attribution ---")
    driver_set = traj.driver_agents or set()
    for method, scores in result.attribution.items():
        ranked = sorted(zip(traj.agent_names, scores), key=lambda x: -x[1])
        top_names = [n for n, _ in ranked[: max(1, len(driver_set))]]
        hit = len(set(top_names) & driver_set) if driver_set else None
        marker = f"  (top-{len(top_names)} overlap with true drivers: {hit}/{len(driver_set)})" if driver_set else ""
        print(f"{method:>16s}: " + ", ".join(f"{n}={v:.3f}" for n, v in ranked) + marker)
    print()

    if result.intervention is not None:
        print("--- Stage 3: Intervention (causal validation) ---")
        iv = result.intervention
        print(f"ablated top-attributed agent: {iv.agent_name} (mode={iv.mode})")
        print(f"  baseline collective signal: {iv.baseline_signal:.5f}")
        print(f"  after ablating {iv.agent_name}: {iv.ablated_signal:.5f}  (drop: {iv.signal_drop:.1%})")
        print(f"  after ablating control agent {iv.control_agent}: {iv.control_signal:.5f}  "
              f"(drop: {iv.control_drop:.1%})")
        print(f"  prediction validated: {iv.validated}")
        print()

    kte_path = os.path.join(args.out_dir, f"{args.scenario}_kte_timeline.png")
    graph_path = os.path.join(args.out_dir, f"{args.scenario}_influence_graph.png")
    bars_path = os.path.join(args.out_dir, f"{args.scenario}_attribution_bars.png")

    fig = plot_kte_timeline(result.kte, transitions=result.transitions,
                             shift_step=(traj.shift_step - result.window) // result.step if traj.shift_step else None)
    fig.savefig(kte_path, dpi=120)

    static_W = result.te_mats.mean(axis=-1)
    fig = plot_influence_graph(static_W, traj.agent_names, driver_agents=traj.driver_agents)
    fig.savefig(graph_path, dpi=120)

    fig = plot_attribution_bars(result.attribution, traj.agent_names, driver_agents=traj.driver_agents)
    fig.savefig(bars_path, dpi=120)

    print(f"saved plots to {args.out_dir}/: "
          f"{os.path.basename(kte_path)}, {os.path.basename(graph_path)}, {os.path.basename(bars_path)}")


if __name__ == "__main__":
    main()
