# GRO821

## Setup

```console
uv sync
```

> To install [`uv`](https://docs.astral.sh/uv/), see [this link](https://docs.astral.sh/uv/getting-started/installation/).

## Use

To run the script for the first assignment:

```console
uv run sandbox
```

## Collision benchmark

The sandbox benchmark compares the naive scan, kd-tree, space quadtree,
point quadtree, and compressed quadtree over identical obstacle and robot
workloads. Experiment selection is configured with `CollisionExperiment` in
`src/gro821/sandbox/main.py`.

Each configured run records build time, collision-query time, retained
storage, and correctness for every method and sample. The naive method has no
applicable build metric; its build-time CSV field is empty.

The default command writes the following files under `results/`:

```text
results/<experiment>_samples.csv
results/<experiment>_summary.csv
```

The samples file contains one row per method and sample:

| Column | Meaning |
| --- | --- |
| `method` | Method label |
| `sample_index` | Workload index |
| `build_time_ns` | Structure construction time, in nanoseconds |
| `query_time_ns` | Collision query time, in nanoseconds |
| `storage_bytes` | Retained Python object-graph size |
| `correct` | Result compared with the naive collision result |
| `sample_*` | Workload or method metadata |
| `experiment_*` | Shared experiment configuration |

The summary file contains one row per method with `count`, `mean`, `median`,
`p99`, and `standard_deviation` columns for each measured metric.

## Memory semantics

Storage uses a recursive `sys.getsizeof` object-graph walk. Referenced
containers, nodes, bounds, and points are included, while shared objects are
counted once. This measures retained Python memory for the built structure;
it does not measure temporary allocations made during construction and is not
a portable serialized-size format.

The benchmark also opens a combined matplotlib dashboard for build time,
query time, and storage. Error bars represent the sample standard deviation
reported by the result summaries.
