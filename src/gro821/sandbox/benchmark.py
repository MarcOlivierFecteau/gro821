from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter_ns

from gro821.sandbox.comparison import BenchmarkSample, ComparisonResults, MethodSpec
from gro821.sandbox.geometry import AABB, OBB, Point, make_aabb_from_circle
from gro821.sandbox.memory import retained_size
from gro821.sandbox.robot import (
    RobotConfig,
    conf_is_valid_naive,
    get_link_aabb,
    get_link_obb,
    get_navigable_area,
)
from gro821.sandbox.structures import kdtree as kd
from gro821.sandbox.structures import quadtree as qd
from gro821.sandbox.world import World

Scalar = str | int | float | bool | None


def _collision_boxes(robot: RobotConfig, bb_type: str) -> tuple[AABB | OBB, AABB | OBB]:
    if bb_type == "aabb":
        return get_link_aabb(robot, 0), get_link_aabb(robot, 1)
    return get_link_obb(robot, 0), get_link_obb(robot, 1)


def _indexed_query(
    query: Callable[[object, object], bool],
    structure: object,
    robot: RobotConfig,
    bb_type: str,
) -> bool:
    first, second = _collision_boxes(robot, bb_type)
    return not (query(structure, first) or query(structure, second))


def collision_methods(
    *,
    bb_type: str = "aabb",
    capacity: int = 1,
    include_naive: bool = True,
    include_kdtree: bool = True,
    include_quadtrees: bool = True,
) -> list[MethodSpec]:
    """Return named collision methods sharing the same point/region workload."""
    bb_type = bb_type.lower()
    if bb_type not in {"aabb", "obb"}:
        raise ValueError("`bb_type` must be either 'aabb' or 'obb'.")
    if capacity <= 0:
        raise ValueError("`capacity` must be positive.")

    methods: list[MethodSpec] = []
    if include_naive:
        methods.append(
            MethodSpec(
                name="naive",
                build=lambda points, _region: points,
                query=lambda points, robot: conf_is_valid_naive(robot, points),
                metadata={"bb_type": bb_type},
            )
        )

    def add_indexed_method(
        name: str,
        build: Callable[[list[Point], AABB, int], object],
        query: Callable[[object, AABB | OBB], bool],
        post_build: Callable[[object], object] | None = None,
    ) -> None:
        methods.append(
            MethodSpec(
                name=name,
                build=lambda points, region, build=build: build(points, region, capacity),
                query=lambda structure, robot, query=query: _indexed_query(
                    query, structure, robot, bb_type
                ),
                post_build=post_build,
                metadata={"bb_type": bb_type, "capacity": capacity},
            )
        )

    if include_kdtree:
        add_indexed_method("kd-tree", kd.build_kdtree, kd.kdtree_collides_bb)  # pyright: ignore[reportArgumentType]
    if include_quadtrees:
        add_indexed_method(
            "space-quadtree",
            qd.build_space_quadtree,
            qd.space_quadtree_collides_aabb
            if bb_type == "aabb"
            else qd.space_quadtree_collides_obb,  # pyright: ignore[reportArgumentType]
        )
        add_indexed_method(
            "point-quadtree",
            qd.build_point_quadtree,
            qd.point_quadtree_collides_aabb
            if bb_type == "aabb"
            else qd.point_quadtree_collides_obb,  # pyright: ignore[reportArgumentType]
        )
        add_indexed_method(
            "compressed-quadtree",
            qd.build_space_quadtree,
            qd.compressed_quadtree_collides_aabb
            if bb_type == "aabb"
            else qd.compressed_quadtree_collides_obb,  # pyright: ignore[reportArgumentType]
            post_build=qd.compress_quadtree,  # pyright: ignore[reportArgumentType]
        )
    return methods


def collision_workloads(
    points_and_robots: Iterable[tuple[list[Point], AABB, RobotConfig]],
) -> list[BenchmarkWorkload]:
    """Adapt generated points, search regions, and robot configurations to workloads."""
    workloads: list[BenchmarkWorkload] = []
    for sample_index, (points, region, robot) in enumerate(points_and_robots):
        workloads.append(
            BenchmarkWorkload(
                sample_index=sample_index,
                build_args=(points, region),
                query_args=(robot,),
                expected=conf_is_valid_naive(robot, points),
                metadata={"point_count": len(points)},
            )
        )
    return workloads


def generate_collision_workloads(
    world: World,
    sample_count: int,
    clusters: int,
    points_per_cluster: int,
) -> list[BenchmarkWorkload]:
    """Generate identical obstacle/configuration workloads for all methods."""
    generated: list[tuple[list[Point], AABB, RobotConfig]] = []
    for _ in range(sample_count):
        initial_robot = RobotConfig(
            Point(world.width / 2, world.height / 2),
            min(world.width, world.height) / 15,
            min(world.width, world.height) / 4,
            min(world.width, world.height) / 6,
            0.0,
            0.0,
        )
        robot = RobotConfig(
            initial_robot.base,
            initial_robot.arm_width,
            initial_robot.arm1_length,
            initial_robot.arm2_length,
            world.rng.uniform(-math.pi, math.pi),
            world.rng.uniform(-math.pi, math.pi),
        )
        points = world.generate_obstacles(
            clusters,
            points_per_cluster,
            1,
            max(world.width, world.height) / 2,
            initial_robot,
        )
        region = make_aabb_from_circle(get_navigable_area(robot)[1])
        generated.append((points, region, robot))
    return collision_workloads(generated)


def run_collision_benchmark(
    points_and_robots: Iterable[tuple[list[Point], AABB, RobotConfig]],
    *,
    bb_type: str = "aabb",
    capacity: int = 1,
    include_naive: bool = True,
    include_kdtree: bool = True,
    include_quadtrees: bool = True,
) -> ComparisonResults:
    """Run all selected collision methods over identical generated workloads."""
    return benchmark_methods(
        collision_methods(
            bb_type=bb_type,
            capacity=capacity,
            include_naive=include_naive,
            include_kdtree=include_kdtree,
            include_quadtrees=include_quadtrees,
        ),
        collision_workloads(points_and_robots),
    )


@dataclass(frozen=True)
class BenchmarkWorkload:
    """One generated workload for a benchmarked collision-detection method."""

    sample_index: int
    build_args: tuple[object, ...] = ()
    query_args: tuple[object, ...] = ()
    expected: bool | None = None
    metadata: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("`sample_index` must be non-negative.")
        object.__setattr__(self, "metadata", dict(self.metadata))


def generate_workloads(
    sample_count: int,
    *,
    build_factory: Callable[[int], tuple[object, ...]],
    query_factory: Callable[[int], tuple[object, ...]] | None = None,
    expected_factory: Callable[[int], bool | None] | None = None,
    metadata_factory: Callable[[int], Mapping[str, Scalar]] | None = None,
) -> list[BenchmarkWorkload]:
    """Create a list of benchmark workloads with deterministic sample indices."""
    if sample_count < 0:
        raise ValueError("`sample_count` must be non-negative.")

    query_factory = query_factory or (lambda _index: ())
    expected_factory = expected_factory or (lambda _index: None)
    metadata_factory = metadata_factory or (lambda _index: {})

    workloads: list[BenchmarkWorkload] = []
    for sample_index in range(sample_count):
        workloads.append(
            BenchmarkWorkload(
                sample_index=sample_index,
                build_args=build_factory(sample_index),
                query_args=query_factory(sample_index),
                expected=expected_factory(sample_index),
                metadata=metadata_factory(sample_index),
            )
        )
    return workloads


def run_method(
    method: MethodSpec,
    workload: BenchmarkWorkload,
    *,
    validator: Callable[[object, object], bool] | None = None,
) -> BenchmarkSample:
    """Benchmark a single method over one workload and return the raw sample."""
    build_start = perf_counter_ns()
    built = method.build(*workload.build_args)
    if method.post_build is not None:
        built = method.post_build(built)
    build_time_ns = perf_counter_ns() - build_start

    query_start = perf_counter_ns()
    query_result = (
        method.query(built, *workload.query_args) if workload.query_args else method.query(built)
    )
    query_time_ns = perf_counter_ns() - query_start

    storage_bytes = method.storage(built) if method.storage is not None else None
    if storage_bytes is None:
        storage_bytes = retained_size(built)

    correct: bool | None
    if validator is not None:
        correct = bool(validator(built, query_result))
    elif workload.expected is not None:
        correct = bool(query_result == workload.expected)
    else:
        correct = None

    metadata = dict(workload.metadata)
    metadata.update(method.metadata)
    return BenchmarkSample(
        method=method.name,
        sample_index=workload.sample_index,
        build_time_ns=float(build_time_ns),
        query_time_ns=float(query_time_ns),
        storage_bytes=int(storage_bytes),
        correct=correct,
        metadata=metadata,
    )


def benchmark_methods(
    methods: Sequence[MethodSpec],
    workloads: Iterable[BenchmarkWorkload],
    *,
    validator: Callable[[MethodSpec, object, object], bool] | None = None,
) -> ComparisonResults:
    """Run all methods across all workloads and return the raw comparison results."""
    results = ComparisonResults()
    for workload in workloads:
        for method in methods:
            if validator is not None:
                sample = run_method(
                    method,
                    workload,
                    validator=lambda value, answer: validator(method, value, answer),
                )
            else:
                sample = run_method(method, workload)
            results.add(sample)
    return results


def run_benchmark(
    methods: Sequence[MethodSpec],
    workloads: Iterable[BenchmarkWorkload],
    *,
    validator: Callable[[MethodSpec, object, object], bool] | None = None,
) -> ComparisonResults:
    """Convenience alias matching the benchmark-runner terminology used in the assignment."""
    return benchmark_methods(methods, workloads, validator=validator)


def compare_methods(
    methods: Sequence[MethodSpec],
    workloads: Iterable[BenchmarkWorkload],
    *,
    validator: Callable[[MethodSpec, object, object], bool] | None = None,
) -> ComparisonResults:
    """Alias for the public comparison entry point."""
    return benchmark_methods(methods, workloads, validator=validator)


__all__ = [
    "BenchmarkWorkload",
    "benchmark_methods",
    "collision_methods",
    "collision_workloads",
    "compare_methods",
    "generate_collision_workloads",
    "generate_workloads",
    "run_benchmark",
    "run_collision_benchmark",
    "run_method",
]
