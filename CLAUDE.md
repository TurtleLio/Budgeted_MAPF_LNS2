# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal in this directory

The reason Claude was brought to this directory is to turn it into a **published git repository following best git practices** — proper history, `.gitignore`, commit hygiene, README, license, branching, etc. **Do not modify the source code itself**; the work here is purely about packaging the existing code into a clean, shareable repo.

## What this is

Research code for **Budgeted Multi-Agent Path Finding (MAPF)**: a hybrid solver that combines **LNS2** (Large Neighborhood Search 2 using SIPPS) with **PIBT** (Priority Inheritance with Backtracking), constrained by a per-agent / shared *node budget*. The main contribution is the rich combinatorial set of *budget-distribution strategies* (how the global pool of path-extension nodes is split across agents and across LNS neighbourhoods), evaluated on standard Sturtevant MAPF benchmarks (random / room / maze 32x32).

No `README.md`, `requirements.txt`, `pyproject.toml`, or test suite exists. The code is run as research scripts.

## Python dependencies

`numpy`, `matplotlib`. No `requirements.txt`. Top-level modules `functions_plotting` (provides `plot_step_in_env`) and `valid_check` (provides `validate_solution`) are imported by `run_single_MAPF_func.py` and several files under `algs/` — both live in the repo root.

## Running experiments

Both entry-point scripts hardcode a working assumption that the CWD is **one level above** this directory — `save_test*` writes to `../results/...` and `run_single_MAPF_func.py` reads `../maps`, `../logs_for_heuristics`. Run from the parent dir or arrange equivalent relative paths.

**Driver script** (`run_test_LNS2_PIBT_excess_budget.py`) — CLI with 9 positional args; loops over scenes 1..25:

```
python Budgeted_MAPF_LNS2/run_test_LNS2_PIBT_excess_budget.py \
    <map_name> <number_of_agents> <alg_name> \
    <resource_distribution_name> <neighbourhood_distribution_name> <agents_distribution_name> \
    <resources_per_agent> <prefix> <folder_name>
```

Example: `python run_test_LNS2_PIBT_excess_budget.py random-32-32-10.map 100 oneshot_LNS2_PIBT_LNS1 fixed Pid agents-shared 50 10 prefix`

Distribution names are resolved via `get_distribution_class()`, which scans `__subclasses__()` of `resource_distribution_class`, `neib_resources`, `neib_agents_distribution` in `globals.py`. Add a new strategy by subclassing one of those with the appropriate `distribution_name` / `neib_budget_name` / `neib_agents_name` attribute. Note: if `neighbourhood_distribution_name` contains `'PIBT'`, the driver overrides it to the `'PIBT'` class — keep that branch in mind when adding new neighbourhood classes whose name happens to contain "PIBT".

**Direct call** (`algs/alg_choise_LNS2_PIBT_excess_budget.py`) — `run_best_of_excess_budget(...)` accepts class objects (not names). The `if __name__ == '__main__':` block has many commented-out parameter sweeps; the active run at the bottom uses a small `4_7_soft_lock.map` that does not ship in `maps/`.

Algorithm name mapping (driver `alg_name` arg → internal `alg_name`):

| CLI arg                          | Internal name                     |
|----------------------------------|-----------------------------------|
| `oneshot_LNS2_PIBT_LNS1`         | `k-LNS2-PIBT-LNS1`                |
| `lifelong_LNS2_PIBT_LNS1`        | `Lifelong-LNS2-PIBT-LNS1`         |
| `oneshot_LNS2_PIBT_spillover`    | `k-LNS2-PIBT-spillover`           |
| `oneshot_LNS2_PIBT_LNS1-pid`     | `k-LNS2-PIBT-LNS1-pid`            |
| `oneshot_LNS2_PIBT_LNS1-DPB`     | `k-LNS2-PIBT-LNS1-DPB`            |
| `oneshot_LNS2_PIBT`              | `k-LNS2-PIBT`                     |
| (and `lifelong_` variants)       | `Lifelong-...`                    |

Substring matching on this name (`'k-LNS2-PIBT'`, `'Lifelong-LNS2-PIBT'`, `'pid'`, `'DPB'`, `'spillover'`, `'LNS1'`) drives termination conditions, output paths, and LNS1 selection branches throughout the code. Renaming requires updating every `in params['alg_name']` check.

**Output routing** (`save_test`, `save_test_LNS2`, `save_test_PIE` in `functions_general.py`) — the `folder_name` arg is *substring-matched* against `number`, `resources`, `prefix`, `max_budget`, `LNS1_none`, `LNS1_shared`, `LNS1_pid`; matching keywords write JSON to different `../results/<sub-dir>/` trees. A run with a `folder_name` that matches **no** keyword saves nothing silently.

## Architecture

### Module layout

- `globals.py` — All core types (`Node`, `AgentAlg`) **and** the entire distribution class hierarchy. Despite the name, it is *not* just constants — algorithms instantiate distribution classes from here.
- `functions_general.py` — Map loading (`get_np_from_dot_map`), graph build (`build_graph_from_np`), heuristic load (`exctract_h_dict`), constraint tables (`init_constraints`, `init_constraints_pie`, `update_constraints_tracked`), scenario load (`making_start_and_goal_lists` — Sturtevant `.scen` format), goal reassignment for lifelong (`update_goal_nodes`), result writers (`save_test*`), `@use_profiler` decorator that dumps `cProfile` stats to `../stats/`.
- `run_single_MAPF_func.py` — Thin harness: load map+scen → build graph+h_dict → call `alg(start, goal, nodes, ...)` → save.
- `algs/alg_choise_LNS2_PIBT_excess_budget.py` — Top-level hybrid alg. Two entry functions: `run_choice_alg_improve_LNS1` (default), `run_choice_alg_spillover`. Both loop over `n_steps`; each step plans a k-prefix with LNS2 *and* PIBT, picks the better by `calculate_state_score` (lexicographic on descending remaining-distances), then optionally consumes leftover budget via `improve_LNS1_shared`.
- `algs/alg_functions_LNS2.py` — LNS2 internals: subset selection (`get_k_limit_agents_subset`, `get_agents_subset_lns`), conflict-pair graph (`get_k_limit_cp_graph`), PrP repair (`solve_subset_with_prp`, `solve_k_limit_subset_with_prp`).
- `algs/alg_prefix_PIBT.py` + `algs/real_alg_functions_pibt.py` — PIBT priority-inheritance solver, truncated at `k_limit`.
- `algs/alg_sipps.py` + `algs/alg_sipps_functions.py` — SIPPS (Safe-Interval PPS) single-agent path finder used by LNS2 for re-planning.
- `algs/alg_limited_temporal_a_star_neighb.py` + `algs/alg_temporal_a_star_functions.py` — Alternative single-agent temporal A*.
- `algs/alg_functions_PIE.py` — Conflict-graph helpers shared by LNS-family algorithms.

### Budget / distribution class system (the central abstraction)

Three orthogonal base classes in `globals.py`, combined via multiple inheritance into a dynamically-created class:

```python
DynamicClass = type(
    f"{dist.distribution_name}-{neib.neib_budget_name}-{neib_agents.neib_agents_name}",
    (distribution_class, neib_resources_subclass, neib_agents_subclass),
    {'class_name': ...}
)
# Use resource_and_neib_distributions.create(...) — don't construct by hand.
```

Roles:

1. **`resource_distribution_class`** — how the global node pool is allocated to agents up-front. Subclasses: `fixed_distribution` (per-agent quota), `shared_distribution` (one pool indexed at `max_nodes[-1]`), `PIB_distribution` (per-agent + shared remainder).
2. **`neib_resources`** — for each LNS neighbourhood, how many nodes the subset is allowed to consume this round. Subclasses include `neib_shared` (all of bank), `neib_proportions_with_prefix` (= `Conflict-Proportional-Budget`), `neib_proportions_with_prefix_reversed` (= `Reversed-Conflict-Proportional-Budget`), `neib_pid` / `neib_pid_distance` / `neib_pid_pie` (PID-controller style — `pid=[kp,kd,ki]`), `neib_multi_arm_bandit`, `neib_distance_proportions` (= `Distance-Proportional-Budget`), `neib_fixed_50/100/150`.
3. **`neib_agents_distribution`** — within a neighbourhood, how the granted pool is split across its agents. Subclasses: `neib_agents_shared`, `neib_agents_evenly_split`, `neib_agents_portion`.

The contract every distribution must respect: `max_nodes` is an `np.ndarray` of length `num_agents + 1`; the `[-1]` slot is the global remainder bank that `resource_collection()` builds up. Distribution names (registered via the class attributes `distribution_name` / `neib_budget_name` / `neib_agents_name`) double as identifiers in both CLI args and saved JSON.

### Core data types

- **`Node`** (`globals.py`) — grid cell with `(x, y)`, `xy_name = f'{x}_{y}'`, list of neighbour `Node`s. `__hash__` keys off `xy_name`. Note: the property `get_fast_heuristic` references `self._heuristic_cache` but `__init__` sets `self.heuristic_cache` (different attribute) — calling that method will raise; only `heuristic_cache` is safe to use directly.
- **`AgentAlg`** (`globals.py`) — has both `path` (committed history extended each outer step) and `k_path` (the current k-prefix being planned). The hybrid loop appends `k_path[1:]` onto `path` once it commits a step. `__eq__`/`__hash__` use `num`.

### The hybrid step loop

In `run_choice_alg_improve_LNS1` (the default for most algorithms via the `else` branch of `run_best_of_excess_budget`):

1. When `step_iter == path_len` (i.e. previous commitment is exhausted), snapshot current positions as new start/goal.
2. Call `solve_k_LNS2` → LNS2 produces `k_path` of length `k_limit+1` for each agent, returns `excess_budget`.
3. Call `run_prefix_pibt` → PIBT produces an alternative k-prefix.
4. `calculate_state_score(LNS_agents, pibt_agents, choise_counter, h_dict)` picks whichever has the lexicographically smaller sorted-descending distance-to-goal vector (ties → LNS).
5. If `'LNS1' in alg_name` and `excess_budget > 0`, `improve_LNS1_shared` runs LNS1 repair on the chosen prefix to spend the leftover, possibly improving SoC.
6. Commit the chosen `k_path` onto each agent's `path`, advance.
7. **Termination**: `k-LNS2-PIBT*` (oneshot) stops when all agents reach their goal; `Lifelong-LNS2-PIBT*` runs the full `n_steps` while reassigning goals via `update_goal_nodes`, tracking `throughput`.

The `agent_distances[agent_id]` dict — with `'raw'` (delay to goal) and `'normalized'` slots — is passed into the distribution via `resources.agent_distances` so PID/distance-proportional strategies can read it.

### Data files & scenarios

- `maps/<name>.map` — Sturtevant grid maps (`.` walkable, `@` wall).
- `maps/scenes/<map>-scen-even/scen-even/<map>-even-<i>.scen` — Sturtevant scenarios, 25 per map (indices 1..25). Loaded by `making_start_and_goal_lists`.
- `logs_for_heuristics/h_dict_of_<map-stem>.json` — pre-computed all-pairs shortest-path heuristic, keyed by goal `xy_name`. **Required**; `exctract_h_dict` raises if missing. There is no on-the-fly fallback in this repo.
- `results/combined_results.json` — example combined output (one record per run).

A new map needs both a `.map` file and a matching `h_dict_of_*.json` heuristic file generated externally — there is no heuristic-builder script in this directory.
