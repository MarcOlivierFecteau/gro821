#!/usr/bin/env python3

import time
from dataclasses import dataclass
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt

import gro821.sandbox.structures.quadtree as qd
from gro821.sandbox import fmt
from gro821.sandbox.benchmark import (
    benchmark_methods,
    collision_methods,
    generate_collision_workloads,
)
from gro821.sandbox.comparison import ComparisonResults, export_csv
from gro821.sandbox.render.plots import plot_benchmark_dashboard
from gro821.sandbox.world import World


def run_collision_experiment(
    world: World,
    sample_count: int,
    clusters: int,
    points_per_cluster: int,
    *,
    bb_type: str = "aabb",
    capacity: int = 1,
    seed: int | None = None,
) -> ComparisonResults:
    """Run all collision methods over one shared set of generated workloads."""
    workloads = generate_collision_workloads(
        world,
        sample_count,
        clusters,
        points_per_cluster,
    )
    return benchmark_methods(
        collision_methods(bb_type=bb_type, capacity=capacity),
        workloads,
        metadata={
            "world_width": world.width,
            "world_height": world.height,
            "cluster_count": clusters,
            "points_per_cluster": points_per_cluster,
            "sample_count": sample_count,
            "capacity": capacity,
            "bb_type": bb_type.lower(),
            "random_seed": seed,
        },
    )


@dataclass(frozen=True)
class CollisionExperiment:
    """Configuration for one reproducible collision benchmark scenario."""

    name: str
    world_width: float
    world_height: float
    sample_count: int
    clusters: int
    points_per_cluster: int
    bb_type: str = "aabb"
    capacity: int = 1
    seed: int = 0
    output_directory: Path | None = None

    def run(self) -> ComparisonResults:
        results = run_collision_experiment(
            World(self.world_width, self.world_height, rng=Random(self.seed)),
            self.sample_count,
            self.clusters,
            self.points_per_cluster,
            bb_type=self.bb_type,
            capacity=self.capacity,
            seed=self.seed,
        )
        if self.output_directory is not None:
            export_csv(results, self.output_directory, self.name)
        return results


def run_configured_experiments(
    experiments: tuple[CollisionExperiment, ...],
) -> dict[str, ComparisonResults]:
    """Run named experiment configurations and return their raw results."""
    return {experiment.name: experiment.run() for experiment in experiments}


def main() -> None:
    _seed = time.time_ns()
    experiments = (
        CollisionExperiment(
            name="sparse-aabb",
            world_width=50,
            world_height=50,
            sample_count=1000,
            clusters=10,
            points_per_cluster=10,
            bb_type="aabb",
            capacity=1,
            seed=_seed,
        ),
        CollisionExperiment(
            name="sparse-obb",
            world_width=50,
            world_height=50,
            sample_count=1000,
            clusters=10,
            points_per_cluster=10,
            bb_type="obb",
            capacity=1,
            seed=_seed,
        ),
        CollisionExperiment(
            name="small-aabb",
            world_width=3,
            world_height=3,
            sample_count=1000,
            clusters=10,
            points_per_cluster=10,
            bb_type="aabb",
            capacity=1,
            seed=_seed,
        ),
    )
    results_by_name = run_configured_experiments(experiments)
    for name, results in results_by_name.items():
        export_csv(results, Path("results"), name)
        print(fmt.bold(f"=== {name} ==="))
        for summary in results.summaries():
            query = summary.query_time_ns
            storage = summary.storage_bytes
            query_mean = f"{query.mean:.0f} ns" if query is not None else "n/a"
            storage_mean = f"{storage.mean:.0f} B" if storage is not None else "n/a"
            print(f"{summary.method}: query = {query_mean}, storage = {storage_mean}")
        plot_benchmark_dashboard(
            results,
            error_bars=True,
            title=f"Benchmark - {name}",
        )
    plt.show(block=True)


if __name__ == "__main__":
    main()
