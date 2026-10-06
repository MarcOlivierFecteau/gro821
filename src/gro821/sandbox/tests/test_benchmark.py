from __future__ import annotations

import math

from gro821.sandbox.benchmark import (
    BenchmarkWorkload,
    benchmark_methods,
    collision_methods,
    collision_workloads,
    generate_workloads,
)
from gro821.sandbox.comparison import MethodSpec
from gro821.sandbox.fmt import ANSI
from gro821.sandbox.geometry import AABB, Point
from gro821.sandbox.robot import RobotConfig


def test_generate_workloads_creates_expected_indices() -> None:
    workloads = generate_workloads(
        3,
        build_factory=lambda index: (index,),
        query_factory=lambda index: (index * 2,),
        expected_factory=lambda index: index % 2 == 0,
        metadata_factory=lambda index: {"index": index},
    )

    assert len(workloads) == 3
    assert workloads[0].sample_index == 0
    assert workloads[0].build_args == (0,)
    assert workloads[0].query_args == (0,)
    assert workloads[0].expected is True
    assert workloads[0].metadata == {"index": 0}


def test_benchmark_methods_records_build_query_and_storage() -> None:
    method = MethodSpec(
        name="adder",
        build=lambda value: value,
        query=lambda value, expected: value == expected,
        storage=lambda value: 17,
    )
    workload = BenchmarkWorkload(
        sample_index=1,
        build_args=(42,),
        query_args=(42,),
        expected=True,
        metadata={"tag": "demo"},
    )

    results = benchmark_methods([method], [workload])
    sample = results.samples[0]

    assert sample.method == "adder"
    assert sample.sample_index == 1
    assert sample.build_time_ns is not None
    assert sample.query_time_ns is not None
    assert sample.storage_bytes == 17
    assert sample.correct is True
    assert sample.metadata["tag"] == "demo"


def test_benchmark_methods_accepts_no_expected_value() -> None:
    method = MethodSpec(
        name="identity",
        build=lambda value: value,
        query=lambda value: value,
    )
    workload = BenchmarkWorkload(sample_index=0, build_args=(7,))

    results = benchmark_methods([method], [workload])

    assert results.samples[0].correct is None


def test_collision_runner_matches_naive_result_for_indexed_methods() -> None:
    robot = RobotConfig(Point(0.0, 0.0), 0.1, 1.0, 1.0, 0.0, 0.0)
    region = AABB(Point(-1.0, -1.0), Point(3.0, 3.0))
    workloads = collision_workloads(
        [
            ([Point(0.5, 0.0), Point(2.0, 2.0)], region, robot),
            ([Point(2.0, 2.0), Point(-0.5, 2.0)], region, robot),
        ]
    )

    results = benchmark_methods(collision_methods(bb_type="aabb", capacity=1), workloads)

    assert {sample.method for sample in results.samples} == {
        "naive",
        "kd-tree",
        "space-quadtree",
        "point-quadtree",
        "compressed-quadtree",
    }
    for sample in results.samples:
        assert sample.correct is True


def test_collision_runner_checks_obb_variants_against_naive() -> None:
    robot = RobotConfig(Point(0.0, 0.0), 0.1, 1.0, 1.0, 0.2, -0.4)
    region = AABB(Point(-1.0, -1.0), Point(3.0, 3.0))
    workloads = collision_workloads(
        [
            ([Point(0.5 * math.cos(0.2), 0.5 * math.sin(0.2)), Point(2.5, 2.5)], region, robot)
        ]
    )

    results = benchmark_methods(collision_methods(bb_type="obb", capacity=1), workloads)

    assert all(sample.correct is True for sample in results.samples)


def main():
    test_generate_workloads_creates_expected_indices()
    test_benchmark_methods_records_build_query_and_storage()
    test_benchmark_methods_accepts_no_expected_value()
    test_collision_runner_matches_naive_result_for_indexed_methods()
    test_collision_runner_checks_obb_variants_against_naive()

    print(f"Benchmark: {ANSI.BRIGHT_GREEN}All tests passed.{ANSI.RESET}")


if __name__ == "__main__":
    main()
