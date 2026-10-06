#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter_ns

import matplotlib.pyplot as plt

import gro821.sandbox.structures.quadtree as qd
from gro821.sandbox import fmt
from gro821.sandbox.algorithms import convex_hull as CH
from gro821.sandbox.benchmark import (
    benchmark_methods,
    collision_methods,
    generate_collision_workloads,
)
from gro821.sandbox.comparison import ComparisonResults, TimeResults, export_csv
from gro821.sandbox.fmt import ANSI
from gro821.sandbox.geometry import Point, make_aabb_from_circle
from gro821.sandbox.render import matplotlib_renderer as renderer
from gro821.sandbox.render.plots import plot_benchmark_dashboard
from gro821.sandbox.robot import (
    RobotConfig,
    conf_is_valid_naive,
    generate_rand_conf_prop_world,
    get_link_aabb,
    get_link_obb,
    get_navigable_area,
)
from gro821.sandbox.structures import kdtree as kd
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


def compare_convex_hull(
    world: World, sample_size: int, num_points: int
) -> tuple[TimeResults, list[Point]]:
    results: list[TimeResults] = []
    convex_hull: list[Point] = []
    for _ in range(sample_size):
        points = world.generate_points(num_points)
        slow_start = perf_counter_ns()
        _ = CH.slow_convex_hull(points, "ccw")
        slow_end = perf_counter_ns()
        start = perf_counter_ns()
        convex_hull = CH.convex_hull(points)
        end = perf_counter_ns()
        rel_time = (end - start) / (slow_end - slow_start)
        speed_factor = 1 / rel_time
        results.append(TimeResults(slow_end - slow_start, end - start, speed_factor))
    avg_slow = avg_time = 0
    avg_factor = 0.0
    for slow_time, time, factor in results:
        avg_slow += slow_time
        avg_time += time
        avg_factor += factor
    avg_slow /= sample_size
    avg_time /= sample_size
    avg_factor /= sample_size
    print(f"Average Performance of Convex Hull\n({sample_size} samples of {num_points} points)")
    print(f"SlowConvexHull: {avg_slow} ns")
    print(f"ConvexHull: {avg_time} ns ({(avg_factor):.3f}x faster)")
    return (TimeResults(avg_slow, avg_time, avg_factor), convex_hull)


def compare_quadtree_to_naive(
    world: World,
    sample_size: int,
    num_points: int,
    bb_type: str,
    split_method: str,
    compressed: bool = False,
    robot: RobotConfig | None = None,
) -> tuple[TimeResults, list[Point]]:
    if bb_type.lower() not in {"aabb", "obb"}:
        raise ValueError("`bb_type` must be either 'aabb' or 'obb'.")
    if split_method.lower() not in {"point", "space"}:
        raise ValueError("`split_method` must be either 'point' or 'space'.")
    if split_method.lower() == "point" and compressed:
        raise ValueError("Invalid `split_method` for `compressed`.")

    robot_override = robot is not None

    results: list[TimeResults] = []
    if any(
        [
            sample_size <= 0,
            num_points <= 0,
            not isinstance(sample_size, int),
            not isinstance(num_points, int),
        ]
    ):
        raise ValueError("`sample_size` and `num_points` take positive integer values.")

    for _ in range(sample_size):
        # Keep robot roughly proportional to the world
        if not robot_override:
            robot = generate_rand_conf_prop_world(world)
        points = world.generate_points(num_points)
        robot_envelope = make_aabb_from_circle(get_navigable_area(robot)[1])  # pyright: ignore[reportArgumentType]
        if split_method.lower() == "point":
            quadtree = qd.build_point_quadtree(
                points, robot_envelope, 1
            )  # Worst case: always split
        else:
            quadtree = qd.build_space_quadtree(
                points, robot_envelope, 1
            )  # Worst case: always split
            if compressed:
                quadtree = qd.compress_quadtree(quadtree)

        if bb_type.lower() == "aabb":
            arm1_bb = get_link_aabb(robot, 0)  # pyright: ignore[reportArgumentType]
            arm2_bb = get_link_aabb(robot, 1)  # pyright: ignore[reportArgumentType]
        else:
            arm1_bb = get_link_obb(robot, 0)  # pyright: ignore[reportArgumentType]
            arm2_bb = get_link_obb(robot, 1)  # pyright: ignore[reportArgumentType]

        naive_start = perf_counter_ns()
        _ = conf_is_valid_naive(robot, points)  # pyright: ignore[reportArgumentType]
        naive_end = perf_counter_ns()

        if bb_type.lower() == "aabb":
            if split_method.lower() == "point":
                quadtree_start = perf_counter_ns()
                _ = qd.point_quadtree_collides_aabb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                _ = qd.point_quadtree_collides_aabb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                quadtree_end = perf_counter_ns()
            else:  # Space
                if compressed:
                    quadtree_start = perf_counter_ns()
                    _ = qd.compressed_quadtree_collides_aabb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                    _ = qd.compressed_quadtree_collides_aabb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                    quadtree_end = perf_counter_ns()
                else:
                    quadtree_start = perf_counter_ns()
                    _ = qd.space_quadtree_collides_aabb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                    _ = qd.space_quadtree_collides_aabb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                    quadtree_end = perf_counter_ns()
        else:  # OBB
            if split_method.lower() == "point":
                quadtree_start = perf_counter_ns()
                _ = qd.point_quadtree_collides_obb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                _ = qd.point_quadtree_collides_obb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                quadtree_end = perf_counter_ns()
            else:  # Space
                if compressed:
                    quadtree_start = perf_counter_ns()
                    _ = qd.compressed_quadtree_collides_obb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                    _ = qd.compressed_quadtree_collides_obb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                    quadtree_end = perf_counter_ns()
                else:
                    quadtree_start = perf_counter_ns()
                    _ = qd.space_quadtree_collides_obb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                    _ = qd.space_quadtree_collides_obb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                    quadtree_end = perf_counter_ns()

        rel_time = (quadtree_end - quadtree_start) / (naive_end - naive_start)
        speed_factor = 1 / rel_time
        results.append(
            TimeResults(naive_end - naive_start, quadtree_end - quadtree_start, speed_factor)
        )

    avg_naive_time = avg_quadtree_time = 0
    avg_factor = 0.0
    for naive_time, quadtree_time, factor in results:
        avg_naive_time += naive_time
        avg_quadtree_time += quadtree_time
        avg_factor += factor
    avg_naive_time /= sample_size
    avg_quadtree_time /= sample_size
    avg_factor /= sample_size

    faster = avg_factor >= 1
    if not faster:
        avg_factor = 1 / avg_factor

    print(fmt.bold("Average Performance of Collision"))
    print(
        fmt.it(f"{sample_size} samples of {num_points} points in ({world.width}, {world.height})")
    )
    if robot_override:
        print(
            f"{ANSI.DIM}Robot: (l: {robot.arm_width}, r2: {robot.arm1_length}, r3: {robot.arm2_length}){ANSI.RESET}"  # pyright: ignore[reportOptionalMemberAccess]
        )

    print(f"Naive approach: {avg_naive_time} ns")

    quadtree_str = f"Quadtree ({split_method.upper()}, {bb_type.upper()}"
    if compressed:
        quadtree_str += ", Compressed"
    quadtree_str += f"): {avg_quadtree_time} ns ({(avg_factor):.3f}x {fmt.pprint_bool(faster, ('faster', 'slower'))})"
    print(quadtree_str)
    fmt.LF()

    return (TimeResults(avg_naive_time, avg_quadtree_time, avg_factor), points)  # pyright: ignore[reportPossiblyUnboundVariable]


def main() -> None:
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
            seed=0,
            output_directory=Path("results"),
        ),
    )
    results_by_name = run_configured_experiments(experiments)
    for name, results in results_by_name.items():
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
