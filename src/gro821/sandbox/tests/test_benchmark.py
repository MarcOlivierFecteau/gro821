from __future__ import annotations

from gro821.sandbox.benchmark import BenchmarkWorkload, benchmark_methods, generate_workloads
from gro821.sandbox.comparison import MethodSpec
from gro821.sandbox.fmt import ANSI


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


def main():
    test_generate_workloads_creates_expected_indices()
    test_benchmark_methods_records_build_query_and_storage()
    test_benchmark_methods_accepts_no_expected_value()

    print(f"Benchmark: {ANSI.BRIGHT_GREEN}All tests passed.{ANSI.RESET}")


if __name__ == "__main__":
    main()
