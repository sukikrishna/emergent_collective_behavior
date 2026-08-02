# Data

CollectiveScope's core pipeline (detection, attribution, intervention) is
dataset-agnostic — it only needs a `Trajectory` (`states` of shape `(T,
n_agents, d)`, plus whatever ground truth is available). Everything in
`collectivescope/datasets/` is an adapter that turns one benchmark's on-disk
format into that shape.

## What's implemented vs. what's a stub

**Fully implemented and tested, no setup required:**

`collectivescope/datasets/synthetic.py` — three generators
(`make_collusion_scenario`, `make_coalition_scenario`, `make_null_scenario`)
with known ground truth (which agents drive the behavior, and when). These
power the demo script, the dashboard's default view, and the entire test
suite in `tests/`. There is no dependency on any external data.

**Implemented against each benchmark's documented log schema, but not yet
run against real benchmark output** — I don't have local copies of these
datasets' actual run logs in this environment, so the parsing logic below
reflects each benchmark's own published schema rather than having been
validated against a real log file. Treat these as a starting point to adapt
once you have real logs, not as verified-working code:

| Adapter | Benchmark | Expected layout under `$DATA_ROOT/` |
|---|---|---|
| `colludebench.py` | ColludeBench-v0 (LLM price-collusion, "Audit the Whisper") | `colludebench/**/*.json` — one JSON per run, with `agents`, `rounds` (per-round per-agent `action`/`reward`), and `colluders` (ground-truth driver set) |
| `werewolf.py` | Werewolf / social-deduction deception benchmark | `werewolf/**/*.json` — one JSON per game, with per-day player messages, a `werewolves` ground-truth set, and vote outcomes |
| `pgg_bench.py` | Public Goods Game benchmark | `pgg_bench/**/*.jsonl` — one JSON-lines event file per game, per-player-per-round contribution amounts |
| `cooperbench.py` | CooperBench (collaborative coding) | `cooperbench/logs/...` — mirrors CooperBench's own on-disk run-log layout; per-step progress + communication-burst signals, `both_passed` eval outcome |

Each adapter exposes the same three functions so they're interchangeable
from the pipeline's point of view:

```python
adapter.is_available()       # True if $DATA_ROOT/<benchmark>/ has data
adapter.load_runs(...)        # or load_games / load_run_dirs — raw records
adapter.build_trajectory(run) # raw record -> Trajectory
```

## Setting DATA_ROOT

```bash
export DATA_ROOT=/path/to/your/benchmark/logs
```

Each adapter looks for its own subdirectory under `DATA_ROOT` (e.g.
`$DATA_ROOT/colludebench/`, `$DATA_ROOT/werewolf/`). If `DATA_ROOT` is unset,
it defaults to `./data/raw` relative to the current working directory.
`data/raw/` is gitignored — real logs are never committed to this repo.

If an adapter's data isn't found, `is_available()` returns `False` and
`load_runs`/`load_games`/`load_run_dirs` raise a `FileNotFoundError` with a
message pointing back here, rather than silently returning an empty list
(which would be easy to mistake for "ran and found nothing collusive").

## Adding a new dataset

Write a function that returns a `Trajectory` (see
`collectivescope/datasets/base.py`). If the benchmark has ground truth about
which agents drive the behavior, set `driver_agents`; if it has a known
onset step, set `shift_step`. That's the entire integration surface — no
other pipeline code needs to change.
