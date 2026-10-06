from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from gro821.sandbox.comparison import ComparisonResults, TimeResults


def plot_metric_summaries(
    results: ComparisonResults,
    *,
    metric: str = "query_time_ns",
    statistic: str = "mean",
    title: str | None = None,
) -> Axes:
    """Plot one summary statistic for every named benchmark method."""
    if metric not in {"build_time_ns", "query_time_ns", "storage_bytes"}:
        raise ValueError("Unsupported benchmark metric.")
    if statistic not in {"count", "mean", "median", "p99", "standard_deviation"}:
        raise ValueError("Unsupported summary statistic.")

    summaries = results.summaries()
    labels = [summary.method for summary in summaries]
    values = []
    for summary in summaries:
        metric_summary = getattr(summary, metric)
        values.append(float(getattr(metric_summary, statistic)) if metric_summary else 0.0)

    figure = plt.figure(title or f"{statistic} {metric}")
    figure.clear()
    axes = figure.add_subplot(111)
    positions = np.arange(len(labels))
    axes.bar(positions, values, width=0.6)
    axes.set_title(title or f"{statistic.title()} {metric.replace('_', ' ')}")
    axes.set_ylabel(metric.replace("_", " ").title())
    axes.set_xticks(positions, labels, rotation=25, ha="right")
    if any(value > 0 for value in values):
        axes.set_yscale("log")
    figure.tight_layout()
    return axes


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
