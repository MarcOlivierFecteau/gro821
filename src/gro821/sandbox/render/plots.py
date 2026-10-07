from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gro821.sandbox.comparison import ComparisonResults, TimeResults

_METRICS = ("build_time_ns", "query_time_ns", "storage_bytes")
_STATISTICS = ("count", "mean", "median", "p99", "standard_deviation")


def save_figure_png(figure: Figure, destination: str | Path) -> Path:
    """Save a matplotlib figure as a tightly-cropped PNG and return its path."""
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png", bbox_inches="tight")
    return output_path


def _summary_values(
    results: ComparisonResults,
    metric: str,
    statistic: str,
) -> tuple[list[str], list[float], list[float]]:
    if metric not in _METRICS:
        raise ValueError("Unsupported benchmark metric.")
    if statistic not in _STATISTICS:
        raise ValueError("Unsupported summary statistic.")

    labels: list[str] = []
    values: list[float] = []
    errors: list[float] = []
    for summary in results.summaries():
        metric_summary = getattr(summary, metric)
        labels.append(summary.method)
        values.append(float(getattr(metric_summary, statistic)) if metric_summary else 0.0)
        errors.append(
            float(metric_summary.standard_deviation) if metric_summary is not None else 0.0
        )
    return labels, values, errors


def _relative_time_values(
    labels: list[str],
    values: list[float],
    errors: list[float],
    baseline_method: str | None,
) -> tuple[list[float], list[float]]:
    if baseline_method is None:
        return values, errors
    if baseline_method not in labels:
        raise ValueError(f"Unknown time baseline method: {baseline_method!r}.")

    baseline = values[labels.index(baseline_method)]
    if baseline <= 0:
        raise ValueError("Time baseline must have a positive value.")
    return (
        [value / baseline for value in values],
        [error / baseline for error in errors],
    )


def _time_axis_limits(
    values: Sequence[float],
    errors: Sequence[float],
    *,
    error_bars: bool,
) -> tuple[float, float] | None:
    bounds = [
        (value - error if error_bars else value, value + error if error_bars else value)
        for value, error in zip(values, errors)
        if value > 0
    ]
    positive_bounds = [bound for pair in bounds for bound in pair if bound > 0]
    if not positive_bounds:
        return None
    lower = min(positive_bounds)
    upper = max(positive_bounds)
    if lower == upper:
        lower /= 2
        upper *= 2
    return lower, upper


def plot_metric_summaries(
    results: ComparisonResults,
    *,
    metric: str = "query_time_ns",
    statistic: str = "mean",
    title: str | None = None,
    error_bars: bool = False,
    time_baseline: str | None = None,
) -> Axes:
    """Plot one summary statistic for every named benchmark method."""
    labels, values, errors = _summary_values(results, metric, statistic)
    if metric in {"build_time_ns", "query_time_ns"}:
        values, errors = _relative_time_values(labels, values, errors, time_baseline)

    figure = plt.figure(title or f"{statistic} {metric}")
    figure.clear()
    axes = figure.add_subplot(111)
    positions = np.arange(len(labels))
    colors = plt.get_cmap("tab10")(np.arange(len(labels)) % 10)
    axes.bar(
        positions,
        values,
        width=0.6,
        color=colors,
        yerr=errors if error_bars else None,
        capsize=4 if error_bars else 0,
    )
    axes.set_title(title or f"{statistic.title()} {metric.replace('_', ' ')}")
    axes.set_ylabel(
        f"Relative {metric.replace('_', ' ')} (baseline = 1)"
        if metric in {"build_time_ns", "query_time_ns"} and time_baseline is not None
        else metric.replace("_", " ").title()
    )
    axes.set_xticks(positions, labels, rotation=25, ha="right")
    if any(value > 0 for value in values):
        axes.set_yscale("log")
    figure.tight_layout()
    return axes


def plot_benchmark_dashboard(
    results: ComparisonResults,
    *,
    statistic: str = "mean",
    error_bars: bool = False,
    title: str = "Benchmark comparison",
    time_baseline: str | None = None,
) -> Figure:
    """Return build, query, and storage plots with optional relative time values."""
    figure, axes = plt.subplots(1, 3, num=title, clear=True, figsize=(15, 5))
    time_data: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for axes_item, metric in zip(axes, _METRICS):
        labels, values, errors = _summary_values(results, metric, statistic)
        if metric in {"build_time_ns", "query_time_ns"}:
            values, errors = _relative_time_values(labels, values, errors, time_baseline)
            time_data[metric] = (labels, values, errors)
        positions = np.arange(len(labels))
        colors = plt.get_cmap("tab10")(np.arange(len(labels)) % 10)
        axes_item.bar(
            positions,
            values,
            width=0.6,
            color=colors,
            yerr=errors if error_bars else None,
            capsize=4 if error_bars else 0,
        )
        axes_item.set_title(metric.replace("_", " ").title())
        axes_item.set_ylabel(
            "Bytes"
            if metric == "storage_bytes"
            else (
                "Relative time (baseline = 1)"
                if time_baseline is not None
                else "Time (ns)"
            )
        )
        axes_item.set_xticks(positions, labels, rotation=25, ha="right")
        if any(value > 0 for value in values):
            axes_item.set_yscale("log")

    time_limits = _time_axis_limits(
        [value for _, values, _ in time_data.values() for value in values],
        [error for _, _, errors in time_data.values() for error in errors],
        error_bars=error_bars,
    )
    if time_limits is not None:
        axes[0].set_ylim(time_limits)
        axes[1].set_ylim(time_limits)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_time_results(results: Sequence[TimeResults], title: str) -> Axes:
    """Plot the legacy two-method time result format."""
    if len(results) not in {2, 4}:
        raise ValueError("Expected two or four TimeResults values.")

    labels = ("Average", "99th percentile")
    collision_results = results[:2]
    figure = plt.figure(f"Collision - {title}")
    figure.clear()
    axes = figure.add_subplot(111)
    positions = np.arange(len(labels))
    width = 0.35
    axes.bar(
        positions - width / 2,
        [result.first for result in collision_results],
        width,
        label="Other",
    )
    axes.bar(
        positions + width / 2,
        [result.second for result in collision_results],
        width,
        label="Kd-tree",
    )
    axes.set_title(f"Collision - {title}")
    axes.set_ylabel("Time (ns)")
    axes.set_yscale("log")
    axes.set_xticks(positions, labels)
    axes.legend()
    figure.tight_layout()
    return axes
