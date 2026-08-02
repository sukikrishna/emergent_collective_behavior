"""Matplotlib helpers for the three pipeline stages. Kept separate from the
Streamlit dashboard so the same functions serve the demo script's saved PNGs
and any future report generation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_kte_timeline(kte_result: dict, transitions: list[int] | None = None, shift_step: int | None = None, ax=None):
    """Detection view: TE level + kinematic decomposition (velocity,
    acceleration, jerk) + the combined KTE norm, with any detected
    transitions and (if known) the ground-truth shift step marked."""
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    else:
        axes = ax

    t = np.arange(len(kte_result["te"]))
    axes[0].plot(t, kte_result["te"], color="#999", alpha=0.5, label="raw TE")
    axes[0].plot(t, kte_result["te_smooth"], color="#1f77b4", label="TE (smoothed)")
    axes[0].set_ylabel("Transfer Entropy")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(t, kte_result["kte"], color="#d62728", label="KTE norm")
    axes[1].set_ylabel("KTE (kinematic norm)")
    axes[1].set_xlabel("timestep")
    axes[1].legend(loc="upper left", fontsize=8)

    if shift_step is not None:
        for a in axes:
            a.axvline(shift_step, color="green", linestyle="--", alpha=0.7, label="true shift")
    if transitions:
        for tr in transitions:
            for a in axes:
                a.axvline(tr, color="red", linestyle=":", alpha=0.8)

    plt.tight_layout()
    return axes[0].figure


def plot_influence_graph(W: np.ndarray, agent_names: list[str], driver_agents: set[str] | None = None, ax=None):
    """Attribution view: directed graph, edge width = TE strength, driver
    agents (if known) highlighted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    n = len(agent_names)
    G = nx.DiGraph()
    for name in agent_names:
        G.add_node(name)
    max_w = W.max() + 1e-12
    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] > 0.01 * max_w:
                G.add_edge(agent_names[i], agent_names[j], weight=W[i, j])

    pos = nx.circular_layout(G)
    driver_agents = driver_agents or set()
    node_colors = ["#d62728" if name in driver_agents else "#1f77b4" for name in G.nodes()]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_edge = max(weights) if weights else 1.0
    widths = [1 + 4 * (w / max_edge) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#555", arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax)
    ax.set_title("Influence graph (red = ground-truth driver)")
    ax.axis("off")
    return ax.figure


def plot_attribution_bars(scores: dict[str, np.ndarray], agent_names: list[str], driver_agents: set[str] | None = None, ax=None):
    """Attribution view: per-method bar chart of per-agent scores, so
    different attribution metrics (out-strength, PageRank, KCA, ...) can be
    compared side by side."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    driver_agents = driver_agents or set()
    n_methods = len(scores)
    n_agents = len(agent_names)
    x = np.arange(n_agents)
    width = 0.8 / max(n_methods, 1)

    for k, (method, vals) in enumerate(scores.items()):
        norm = vals / (vals.max() + 1e-12)
        colors = ["#d62728" if name in driver_agents else "#1f77b4" for name in agent_names]
        bars = ax.bar(x + k * width, norm, width=width, label=method, alpha=0.5 + 0.5 * (k == 0))
        if k == 0:
            for bar, c in zip(bars, colors):
                bar.set_edgecolor(c)
                bar.set_linewidth(2)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(agent_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("normalized score")
    ax.legend(fontsize=8)
    ax.set_title("Attribution scores by method (red outline = ground-truth driver)")
    plt.tight_layout()
    return ax.figure
