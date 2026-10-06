from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median, pstdev

type Scalar = str | int | float | bool | None
type BuildFunction = Callable[..., object]
type QueryFunction = Callable[..., bool]
type TransformFunction = Callable[[object], object]
type StorageFunction = Callable[[object], int]


@dataclass
class TimeResults:
    first: float
    second: float
    factor: float

    def __iter__(self):
        return (self.first, self.second, self.factor).__iter__()


@dataclass
class ComparisonConfig:
    structure: str
    split_method: str | None = None  # used for quadtrees
    bb_type: str | None = None
    compressed: bool | None = None  # used for quadtrees
    capacity: int = 1

    def __post_init__(self) -> None:
        self.structure = self.structure.lower()
        if self.split_method:
            self.split_method = self.split_method.lower()
        if self.bb_type:
            self.bb_type = self.bb_type.lower()
        if self.capacity <= 0:
            raise ValueError("`capacity` must be positive.")


@dataclass
class MethodSpec:
    """Describe one collision-detection method for a benchmark run."""

    name: str
    build: BuildFunction
    query: QueryFunction
    storage: StorageFunction | None = None
    post_build: TransformFunction | None = None
    measure_build: bool = True
    config: ComparisonConfig | None = None
    metadata: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("`name` must not be empty.")
        self.metadata = dict(self.metadata)


@dataclass(frozen=True)
class BenchmarkSample:
    """One method's measurements for one generated workload."""

    method: str
    sample_index: int
    build_time_ns: float | None = None
    query_time_ns: float | None = None
    storage_bytes: int | None = None
    correct: bool | None = None
    metadata: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("`method` must not be empty.")
        if self.sample_index < 0:
            raise ValueError("`sample_index` must not be negative.")
        if self.storage_bytes is not None and self.storage_bytes < 0:
            raise ValueError("`storage_bytes` must not be negative.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MetricSummary:
    """Descriptive statistics for one measured metric."""

    count: int
    mean: float
    median: float
    p99: float
    standard_deviation: float

    @classmethod
    def from_values(cls, values: Iterable[float]) -> MetricSummary:
        ordered = sorted(values)
        if not ordered:
            raise ValueError("Cannot summarize an empty collection.")

        index = (len(ordered) - 1) * 0.99
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        p99 = ordered[lower] + (ordered[upper] - ordered[lower]) * weight
        return cls(
            count=len(ordered),
            mean=fmean(ordered),
            median=median(ordered),
            p99=p99,
            standard_deviation=pstdev(ordered),
        )


@dataclass(frozen=True)
class MethodSummary:
    """Aggregated measurements for one named method."""

    method: str
    build_time_ns: MetricSummary | None
    query_time_ns: MetricSummary | None
    storage_bytes: MetricSummary | None


@dataclass
class ComparisonResults:
    """Raw benchmark observations and experiment-level metadata."""

    samples: list[BenchmarkSample] = field(default_factory=list)
    metadata: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata)

    def add(self, sample: BenchmarkSample) -> None:
        self.samples.append(sample)

    def summaries(self) -> list[MethodSummary]:
        methods: list[str] = []
        for sample in self.samples:
            if sample.method not in methods:
                methods.append(sample.method)

        summaries = []
        for method in methods:
            method_samples = [sample for sample in self.samples if sample.method == method]
            summaries.append(
                MethodSummary(
                    method=method,
                    build_time_ns=_summarize(sample.build_time_ns for sample in method_samples),
                    query_time_ns=_summarize(sample.query_time_ns for sample in method_samples),
                    storage_bytes=_summarize(sample.storage_bytes for sample in method_samples),
                )
            )
        return summaries


def _summarize(values: Iterable[float | None]) -> MetricSummary | None:
    present = [value for value in values if value is not None]
    return MetricSummary.from_values(present) if present else None


def write_samples_csv(results: ComparisonResults, destination: str | Path) -> None:
    """Write one row per raw method/sample observation."""
    metadata_keys = sorted({key for sample in results.samples for key in sample.metadata})
    fieldnames = [
        "method",
        "sample_index",
        "build_time_ns",
        "query_time_ns",
        "storage_bytes",
        "correct",
        *[f"sample_{key}" for key in metadata_keys],
        *[f"experiment_{key}" for key in sorted(results.metadata)],
    ]

    with Path(destination).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample in results.samples:
            row: dict[str, Scalar] = {
                "method": sample.method,
                "sample_index": sample.sample_index,
                "build_time_ns": sample.build_time_ns,
                "query_time_ns": sample.query_time_ns,
                "storage_bytes": sample.storage_bytes,
                "correct": sample.correct,
            }
            row.update({f"sample_{key}": sample.metadata.get(key) for key in metadata_keys})
            row.update(
                {f"experiment_{key}": results.metadata[key] for key in sorted(results.metadata)}
            )
            writer.writerow(row)


def write_summary_csv(results: ComparisonResults, destination: str | Path) -> None:
    """Write descriptive statistics for each method and measured metric."""
    metric_names = ("build_time_ns", "query_time_ns", "storage_bytes")
    statistic_names = ("count", "mean", "median", "p99", "standard_deviation")
    fieldnames = ["method"] + [
        f"{metric}_{statistic}" for metric in metric_names for statistic in statistic_names
    ]
    fieldnames.extend(f"experiment_{key}" for key in sorted(results.metadata))

    with Path(destination).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in results.summaries():
            row: dict[str, Scalar] = {"method": summary.method}
            for metric_name in metric_names:
                metric = getattr(summary, metric_name)
                for statistic in statistic_names:
                    row[f"{metric_name}_{statistic}"] = (
                        getattr(metric, statistic) if metric is not None else None
                    )
            row.update(
                {f"experiment_{key}": results.metadata[key] for key in sorted(results.metadata)}
            )
            writer.writerow(row)
