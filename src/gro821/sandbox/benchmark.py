from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter_ns

from gro821.sandbox.comparison import BenchmarkSample, ComparisonResults, MethodSpec
from gro821.sandbox.memory import retained_size

Scalar = str | int | float | bool | None


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
    "compare_methods",
    "generate_workloads",
    "run_benchmark",
    "run_method",
]
