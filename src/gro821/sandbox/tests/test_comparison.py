from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from gro821.sandbox.benchmark import BenchmarkWorkload, benchmark_methods, collision_methods
from gro821.sandbox.comparison import (
    ComparisonConfig,
    MethodSpec,
    MetricSummary,
    export_csv,
)
from gro821.sandbox.fmt import ANSI


def assert_float_equal(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    """Assert that two floating-point values differ by no more than ``tolerance``."""
    if tolerance < 0:
        raise ValueError("`tolerance` must not be negative.")
    difference = abs(actual - expected)
    assert difference <= tolerance, (
        f"Expected {actual!r} to equal {expected!r} within {tolerance}, "
        f"but the difference was {difference}."
    )


def test_metric_summary_computes_expected_statistics() -> None:
    summary = MetricSummary.from_values([1.0, 2.0, 3.0, 4.0])

    assert summary.count == 4
    assert_float_equal(summary.mean, 2.5)
    assert_float_equal(summary.median, 2.5)
    assert_float_equal(summary.p99, 3.97)
    assert summary.standard_deviation > 1.11


def test_export_csv_contains_raw_and_summary_metadata() -> None:
    method = MethodSpec(
        name="identity",
        build=lambda value: value,
        query=lambda value: value == 1,
        storage=lambda _value: 12,
    )
    workloads = [
        BenchmarkWorkload(sample_index=0, build_args=(1,), expected=True, metadata={"size": 1}),
        BenchmarkWorkload(sample_index=1, build_args=(2,), expected=False, metadata={"size": 2}),
    ]
    results = benchmark_methods(
        [method],
        workloads,
        metadata={"world_width": 3.0, "sample_count": 2},
    )

    with TemporaryDirectory() as directory:
        samples_path, summary_path = export_csv(results, directory, "identity")
        with samples_path.open(newline="", encoding="utf-8") as stream:
            sample_rows = list(csv.DictReader(stream))
        with summary_path.open(newline="", encoding="utf-8") as stream:
            summary_rows = list(csv.DictReader(stream))

    assert samples_path == Path(directory) / "identity_samples.csv"
    assert summary_path == Path(directory) / "identity_summary.csv"
    assert len(sample_rows) == 2
    assert sample_rows[0]["sample_size"] == "1"
    assert sample_rows[0]["experiment_world_width"] == "3.0"
    assert sample_rows[0]["correct"] == "True"
    assert len(summary_rows) == 1
    assert summary_rows[0]["query_time_ns_count"] == "2"
    assert summary_rows[0]["storage_bytes_mean"] == "12.0"
    assert summary_rows[0]["experiment_sample_count"] == "2"


def test_invalid_comparison_configurations_are_rejected() -> None:
    try:
        ComparisonConfig("quadtree", capacity=0)
    except ValueError:
        pass
    else:
        raise AssertionError("A non-positive comparison capacity must be rejected.")

    try:
        MethodSpec(name="", build=lambda: None, query=lambda _value: True)
    except ValueError:
        pass
    else:
        raise AssertionError("An empty method name must be rejected.")

    try:
        collision_methods(bb_type="circle")
    except ValueError:
        pass
    else:
        raise AssertionError("An unsupported bounding-box mode must be rejected.")


def main() -> None:
    test_metric_summary_computes_expected_statistics()
    test_export_csv_contains_raw_and_summary_metadata()
    test_invalid_comparison_configurations_are_rejected()
    print(f"Comparison: {ANSI.BRIGHT_GREEN}All tests passed.{ANSI.RESET}")


if __name__ == "__main__":
    main()
