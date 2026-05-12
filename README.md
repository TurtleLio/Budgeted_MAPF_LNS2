# Budgeted MAPF — LNS2 + PIBT

Research code for **Multi-Agent Path Finding (MAPF) under a node budget**. The
solver hybridises **LNS2** (Large Neighborhood Search 2, using SIPPS for
single-agent replanning) with **PIBT** (Priority Inheritance With Backtracking);
at every planning step both methods produce a `k_limit`-length prefix and the
better one is committed. The contribution is the family of **budget-distribution
strategies** that govern how a limited pool of path-extension nodes is split
across agents and across the LNS neighbourhoods chosen for repair.

Benchmarks: the standard Sturtevant `random-32-32-10`, `room-32-32-4` and
`maze-32-32-4` 32 × 32 grids with the canonical `even` scenarios.

## Repository layout

```
.
├── globals.py                       # Core types (Node, AgentAlg) + distribution class hierarchy
├── functions_general.py             # Map / scenario / heuristic / constraint helpers, result writers
├── functions_plotting.py            # Rendering helpers
├── valid_check.py                   # Solution validator
├── run_single_MAPF_func.py          # Per-experiment harness
├── run_test_LNS2_PIBT_excess_budget.py  # CLI driver (loops scenes 1..25)
├── algs/                            # LNS2, PIBT, SIPPS, temporal A*, PIE
├── maps/                            # Sturtevant .map grids + scenes/<map>-scen-even/scen-even/*.scen
├── logs_for_heuristics/             # Pre-computed all-pairs heuristics (h_dict_of_<map>.json)
├── results/                         # Output JSON (sample: combined_results.json)
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.10+ is recommended (the codebase uses `Self`, PEP 604 unions, and the
walrus-friendly typing imports).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running an experiment

The driver script expects to be invoked **from the parent of this directory**
so its `../maps`, `../logs_for_heuristics`, and `../results` paths resolve. From
the repo root, that means running it as:

```bash
cd ..
python Budgeted_MAPF_LNS2/run_test_LNS2_PIBT_excess_budget.py \
    <map> <num_agents> <alg> \
    <resource_dist> <neighbourhood_dist> <agents_dist> \
    <resources_per_agent> <prefix> <folder_name>
```

Example — 100 agents on `random-32-32-10` with the hybrid LNS2 + PIBT solver,
PID-based neighbourhood budgeting, 50 nodes/agent, k-prefix 10:

```bash
python Budgeted_MAPF_LNS2/run_test_LNS2_PIBT_excess_budget.py \
    random-32-32-10.map 100 oneshot_LNS2_PIBT_LNS1 \
    fixed Pid agents-shared \
    50 10 prefix
```

The script loops over scenarios 1..25 and writes a JSON record per scene under
`../results/<sub-folder>/...`. The sub-folder is chosen by substring matching
the `folder_name` argument against a fixed set of keywords (`number`,
`resources`, `prefix`, `max_budget`, `LNS1_none`, `LNS1_shared`, `LNS1_pid`); a
value that matches none is silently saved nowhere.

### Algorithm names (CLI → internal)

| CLI `alg_name`                  | Internal name                |
|---------------------------------|------------------------------|
| `oneshot_LNS2_PIBT_LNS1`        | `k-LNS2-PIBT-LNS1`           |
| `lifelong_LNS2_PIBT_LNS1`       | `Lifelong-LNS2-PIBT-LNS1`    |
| `oneshot_LNS2_PIBT_spillover`   | `k-LNS2-PIBT-spillover`      |
| `oneshot_LNS2_PIBT_LNS1-pid`    | `k-LNS2-PIBT-LNS1-pid`       |
| `oneshot_LNS2_PIBT_LNS1-DPB`    | `k-LNS2-PIBT-LNS1-DPB`       |
| `oneshot_LNS2_PIBT`             | `k-LNS2-PIBT`                |

`oneshot_*` terminates when every agent reaches its goal (success / makespan /
sum-of-costs). `lifelong_*` runs the full `n_steps`, reassigning goals on the
fly, and reports throughput.

### Distribution strategy names

Resource-pool distribution (over all agents, up-front):
`fixed`, `shared`, `PIB`.

Per-neighbourhood budget distribution:
`Shared`, `Conflict-Proportional-Budget`, `Reversed-Conflict-Proportional-Budget`,
`Distance-Proportional-Budget`, `Pid`, `Pid-Distance`, `Pid-Pie`,
`Multi-Arm-Bandit`, `Random`, `Fixed-50`, `Fixed-100`.

Within-neighbourhood split:
`agents-shared`, `agents-evenly-split`, `agent-propotion` (sic).

See `globals.py` for the full hierarchy. Names are registered as class
attributes (`distribution_name`, `neib_budget_name`, `neib_agents_name`) and
resolved at runtime via `__subclasses__()` in
`run_test_LNS2_PIBT_excess_budget.py::get_distribution_class`.

## Maps and scenarios

Map files follow Nathan Sturtevant's `.map` format (`.` walkable, `@` wall) and
scenarios follow the `.scen` even-distribution format from the same benchmark
suite (25 scenarios per map, indexed 1..25). Adding a new map requires both a
`.map` grid in `maps/` and a matching `h_dict_of_<map-stem>.json` all-pairs
heuristic file in `logs_for_heuristics/` — `exctract_h_dict` raises if the
heuristic is missing, and there is no in-repo builder for it.

## License

MIT — see [`LICENSE`](LICENSE).
