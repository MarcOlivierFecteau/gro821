import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from gro821.sandbox.comparison import TimeResults


def plot_time_results(results: list[TimeResults], title: str) -> Axes:
    """Plot average and 99th-percentile collision times as grouped bars."""
    build_times = False
    if len(results) == 2:
        average_results = results[0]
        percentile_results = results[1]
    elif len(results) == 4:
        average_results, percentile_results, average_build_results, percentile_build_results = (
            results
        )
        build_times = True
    else:
        raise ValueError("TODO: better logic.")

    average_speed_factor = average_results.first / average_results.second
    percentile_speed_factor = percentile_results.first / percentile_results.second
    if build_times:
        average_build_speed_factor = average_build_results.first / average_build_results.second  # pyright: ignore[reportPossiblyUnboundVariable]
        percentile_build_speed_factor = (
            percentile_build_results.first / percentile_build_results.second  # pyright: ignore[reportPossiblyUnboundVariable]
        )

    labels = ("Average", "99th percentile")
    other_times = (average_results.first, percentile_results.first)
    kdtree_times = (average_results.second, percentile_results.second)
    if build_times:
        other_build_times = (average_build_results.first, percentile_build_results.first)  # pyright: ignore[reportPossiblyUnboundVariable]
        kdtree_build_times = (average_build_results.second, percentile_build_results.second)  # pyright: ignore[reportPossiblyUnboundVariable]

        build_figure = plt.figure(f"Build - {title}")
        build_figure.clear()
        build_axes = build_figure.add_subplot(111)

        positions = np.arange(len(labels))
        width = 0.35
        build_axes.bar(positions - width / 2, other_build_times, width, label="Other")
        build_axes.bar(positions + width / 2, kdtree_build_times, width, label="Kd-tree")
        for position, height, speed_factor in zip(
            positions + width / 2, kdtree_build_times, (average_build_speed_factor, percentile_build_speed_factor)  # pyright: ignore[reportPossiblyUnboundVariable]
        ):
            build_axes.text(
                position,
                height * 1.08,
                f"{speed_factor:.3f}x",
                ha="center",
                va="bottom",
            )

        build_axes.set_title(f"Build - {title}")
        build_axes.set_ylabel("Time (ns)")
        build_axes.set_yscale("log")
        build_axes.set_xticks(positions, labels)
        build_axes.legend()
        build_figure.tight_layout()

    figure = plt.figure(f"Collision - {title}")
    figure.clear()
    axes = figure.add_subplot(111)

    positions = np.arange(len(labels))
    width = 0.35
    axes.bar(positions - width / 2, other_times, width, label="Other")
    axes.bar(positions + width / 2, kdtree_times, width, label="Kd-tree")
    for position, height, speed_factor in zip(
        positions + width / 2,
        kdtree_times,
        (average_speed_factor, percentile_speed_factor),
    ):
        axes.text(
            position,
            height * 1.08,
            f"{speed_factor:.3f}x",
            ha="center",
            va="bottom",
        )

    axes.set_title(f"Collision - {title}")
    axes.set_ylabel("Time (ns)")
    axes.set_yscale("log")
    axes.set_xticks(positions, labels)
    axes.legend()
    figure.tight_layout()
    return axes
