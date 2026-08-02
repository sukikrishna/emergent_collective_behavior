# CollectiveScope

Detect the onset of emergent collective behavior in multi-agent AI systems,
identify which agents are driving it, and causally validate that
attribution by intervening.

## Why

As AI systems increasingly operate as interacting groups of agents rather
than isolated models, collective behaviors — cooperation, collusion,
deception, consensus — can emerge in ways that are difficult to detect and
even harder to explain. Most existing evaluation is post-hoc: run the
system, see whether the outcome looks collusive/cooperative/etc. That
answers "did something happen" but not "when did it start," "who caused
it," or "would removing that agent actually have stopped it."

CollectiveScope is a three-stage pipeline that tries to answer all three:

1. **Detect** the onset of a collective behavioral shift as early as
   possible, from changes in inter-agent information flow — not just
   classify a whole run after the fact.
2. **Attribute** the behavior to specific agents via a directed influence
   graph — who initiates, propagates, or amplifies it — rather than
   collapsing everything to one system-wide score.
3. **Intervene**: remove, silence, or shuffle the agents attribution says
   matter, rerun, and check whether the collective signal drops the way the
   attribution predicted. This is what turns "agent X looks influential"
   into a falsifiable causal claim.

## How it works

- **Detection** uses Kinematic Transfer Entropy (KTE): transfer entropy
  measures directed information flow between agents, but its *level* stays
  elevated for the whole duration of a regime and so is slow to localize
  *when* something changed. KTE instead tracks the *dynamics* of the TE
  series (its velocity/acceleration/jerk), which spike sharply right at a
  transition.
- **Attribution** builds a full directed (agent × agent) conditional
  transfer-entropy graph — conditioning each pairwise estimate on every
  other agent's state removes common-driver confounds — and reports several
  per-agent metrics (out-strength, in-strength, hub score, causal dominance,
  PageRank, and KCA — the kinematic operator applied per-pair rather than to
  the system-wide average) rather than one scalar ranking.
- **Intervention** ablates the top-attributed agent and a genuinely
  peripheral control agent, then compares how much each ablation hurts the
  collective signal — closing the loop from "looks influential" to
  "causally verified."

See [`docs/design.md`](docs/design.md) for the full design rationale,
including two real bugs found and fixed during development (a detector that
falsely fired on ~65% of no-collusion control runs, and a control-agent
selection bug that made every intervention look "unvalidated") — both are
now guarded against by regression tests.

## Status

This is a first full-pass implementation of the pipeline described above,
covering:

- the core library (`collectivescope/`) — detection, attribution,
  intervention, three synthetic benchmark scenarios with known ground
  truth, and four real-benchmark dataset adapter stubs
- an end-to-end CLI demo (`examples/run_end_to_end_demo.py`)
- an interactive dashboard (`dashboard/app.py`)
- a test suite (`tests/`) that encodes the ground-truth checks described
  above as regression tests

It is **not** yet validated against real benchmark logs (see
[`data/README.md`](data/README.md) for exactly what's implemented vs. what's
a documented stub), and detection thresholds are tuned against synthetic
data's noise characteristics rather than real benchmark data.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the full pipeline on a synthetic scenario and save plots
python examples/run_end_to_end_demo.py --scenario collusion --out-dir outputs

# interactive dashboard
streamlit run dashboard/app.py

# tests
pytest tests/
```

No external data or network access is required for any of the above — the
default data source is the synthetic scenario generator
(`collectivescope/datasets/synthetic.py`). To point the dataset adapters at
real benchmark logs instead, see [`data/README.md`](data/README.md).

## Repo layout

```
collectivescope/
  detection/       transfer entropy (Granger + symbolic), entropy, KTE + transition detection
  attribution/     directed influence graph, per-agent metrics, KCA
  intervention/    agent ablation, causal validation of attribution
  datasets/        Trajectory interface, synthetic generators, real-benchmark adapters
  viz/             matplotlib plots shared by the demo script and dashboard
  pipeline.py      wires the three stages together (run_pipeline)
dashboard/         Streamlit app
examples/          end-to-end CLI demo
tests/             pytest suite, run against the synthetic scenarios' ground truth
docs/design.md     design rationale, mapped to the pipeline's three stages
data/README.md     how to point real-benchmark adapters at DATA_ROOT
```

## Relationship to prior work

The transfer-entropy and kinematic-detection math originates from earlier
prototyping in a separate repo (`~/Documents/KTE`). That repo mixes an
unrelated paper-writing project with large raw datasets and many iterations
of the same experiment script; this repo is an independent, from-scratch
implementation that reuses the underlying ideas (not the code) and adds the
attribution and intervention stages, which the earlier prototyping never
closed the loop on.
