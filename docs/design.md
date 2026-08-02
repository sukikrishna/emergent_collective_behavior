# Design

How the code in `collectivescope/` maps to the three-stage pipeline the
project proposal describes, and the reasoning behind the parts that aren't
obvious from the code alone.

## Stage 1 — Detection (`collectivescope/detection/`)

**Goal**: flag the onset of a collective behavioral shift — not just
classify "collusive vs. honest" after the fact, but localize *when* it
started.

- `transfer_entropy.py` computes directed information flow between every
  pair of agents. Two estimators:
  - **Conditional Granger TE** for continuous states: the log variance-ratio
    between an AR(1) model of agent *j* that includes agent *i*'s lagged
    state and one that doesn't. Conditioning on every other agent's lagged
    state turns this into *conditional* TE, which is what separates "i
    actually influences j" from "i and j are both driven by a third agent."
    Fit with ridge-regularized least squares — the full model always has
    more regressors than the restricted one, so on a short sliding window an
    unregularized fit spuriously lowers the full model's residual variance
    just by having more parameters to chase noise with, inflating TE for
    every pair, not only real senders.
  - **Symbolic (Schreiber) TE** for discrete action/tool-call codes, for
    datasets where states aren't naturally continuous.
- `entropy.py` provides plain Shannon entropy over a sliding window, used
  as a simpler complement/baseline signal.
- `kte.py` is the kinematic operator: it takes the scalar mean-TE time
  series and computes its velocity, acceleration, and jerk (via Gaussian
  smoothing + finite differences), then combines them into one norm. Raw TE
  *level* stays elevated for the whole duration of a collusive regime, so a
  threshold on the level alone fires late and stays high long after — it
  answers "is something happening" but not "when did it start." The
  kinematic norm spikes sharply right at the transition and decays once the
  new regime stabilizes, because it's tracking *change*, not level.
  `detect_transitions` then thresholds that norm to produce timestamps.

**Why the detector needed two fixes during development** (see
`tests/test_detection.py` for the regression tests): a naive
single-window, 3σ-above-burn-in-mean threshold produced false positives on
roughly 65% of seeds of a zero-coupling synthetic control (`make_null_scenario`)
— an unacceptable false-alarm rate for a "nothing is happening" case. Scanning
~90 windows per run with a per-window 3σ test is a multiple-comparisons
problem, and mean/std estimated from a short burn-in slice is itself noisy.
The fix: (1) estimate the baseline from the median/MAD of the *whole* series
(robust to the transition itself, since it's a small fraction of the total
windows) rather than the mean/std of a short burn-in slice, and (2) require
the excursion to persist for several consecutive windows rather than firing
on a single one. That brought the false-positive rate to 0/30 while still
detecting the true onset promptly in the collusion/coalition scenarios.

## Stage 2 — Attribution (`collectivescope/attribution/influence_graph.py`)

**Goal**: identify *who* is driving the collective behavior — not a single
system-wide scalar, but a per-agent map of who initiates, propagates, or
amplifies it.

`InfluenceGraph` wraps the directed (n_agents, n_agents[, T]) conditional-TE
matrix and exposes several metrics that each answer a different question:

- **out-strength** — how much an agent drives everyone else (source of
  coordination).
- **in-strength** — how much an agent is driven (a follower).
- **hub-score** = out-strength × in-strength — agents that both receive and
  relay a lot of information (potential relays/bottlenecks).
- **causal-dominance** — an agent's single strongest outgoing edge relative
  to the average, distinguishing "one disproportionately strong link" from
  "broadly average influence."
- **PageRank** on the row-normalized TE matrix — recursive influence: an
  agent that drives *other high-influence* agents scores higher than one
  that only drives peripheral agents, which plain out-strength can't
  distinguish.

**KCA (Kinematic Causal Attribution)** composes stages 1 and 2: instead of
applying the kinematic operator to the scalar mean-TE series (stage 1), it
applies it to every directed, time-resolved pairwise TE series
independently, then sums each agent's outgoing kinematic flow. This
attributes *who's causal out-flow is changing* at a transition, which
matters when a driver only becomes dominant partway through a run — a
plain time-averaged out-strength would dilute that agent's score with the
independent phase before the shift.

The different metrics deliberately don't always agree, and the demo/tests
reflect that: in the collusion scenario, `out_strength` and
`causal_dominance` (which measure *outgoing* influence) correctly rank only
the leader agent at rank 1, while `in_strength`, `hub_score`, and `pagerank`
(which also credit being on the receiving end of the dominant edge) recover
both the leader and the follower in the top 2. Reporting all of them side by
side, rather than collapsing to one "the" attribution score, is the point —
it's what makes attribution a map rather than a single verdict.

## Stage 3 — Intervention (`collectivescope/intervention/ablation.py`)

**Goal**: close the observe → intervene loop. Attribution's claim ("agent X
is a driver") is only a causal claim if it's falsifiable: removing/silencing
X should collapse the collective signal more than doing the same to a
non-driver.

`ablate_agent` supports three modes: `remove` (drop the agent), `silence`
(replace its trajectory with its own time-mean — a constant series carries
no information, so its outgoing TE should collapse), and `shuffle`
(temporally permute the agent's own trajectory — same marginal statistics,
temporal structure destroyed).

`validate_intervention` ablates the top-attributed agent, ablates a control
agent, and reports whether the top agent's ablation hurt the collective
signal (mean off-diagonal conditional TE) more than the control's did.

**Why control-agent selection needed a fix**: the first version picked the
control as `argmin(attribution_scores)` — the agent the attribution method
itself ranked lowest. That's confounded by network topology, not just
driver-ness: in the collusion scenario, the *follower* driver has the lowest
*outgoing*-influence score (KCA/out-strength measure outgoing flow, and a
follower's outgoing flow is small even though it's a driver by the
scenario's ground truth) while still sitting on the single highest-weight
edge in the graph, as the *target* of the leader. Silencing it collapsed the
edge from both ends, producing a bigger signal drop than silencing the
leader — the intervention wrongly reported "prediction not validated" on
every synthetic scenario. The fix selects the control by **total TE degree**
(in-strength + out-strength from a fresh, full TE matrix) instead of by the
attribution score being validated — an agent that's actually weakly
connected in both directions, which is what "control" is supposed to mean.
See `tests/test_intervention.py::test_control_agent_is_not_the_other_driver`.

## Where the three stages meet: `collectivescope/pipeline.py`

`run_pipeline(trajectory)` is the single entry point the demo script, the
dashboard, and the tests all call — it owns the one set of default
window/step/threshold values so they don't drift between call sites. It
runs detection, builds the influence graph and computes attribution scores,
and (optionally) runs the intervention validation using whichever
attribution method was requested as "primary."

## Known limitations / what a v1 would need

- The real-benchmark dataset adapters (`colludebench.py`, `werewolf.py`,
  `pgg_bench.py`, `cooperbench.py`) are implemented against each benchmark's
  documented schema but have not been run against real log files — see
  `data/README.md`.
- Detection thresholds (`n_std`, `min_run`) were tuned against the synthetic
  scenarios' noise characteristics; real benchmark data will likely need
  re-tuning.
- The intervention loop currently ablates one agent at a time and compares
  against one control; it doesn't yet do the proposal's fuller
  "remove/replace/constrain/steer" set of intervention types, or evaluate
  interventions on subsets of agents.
- No CooperBench-specific evaluation harness yet — the proposal calls out
  CooperBench as a benchmark target, but that adapter builds a `Trajectory`
  without attribution ground truth (CooperBench doesn't label a "driver"
  agent the way the collusion/deception benchmarks do), so it currently only
  exercises the detection stage.
