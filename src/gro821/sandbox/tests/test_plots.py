from matplotlib import pyplot as plt

from gro821.sandbox.comparison import BenchmarkSample, ComparisonResults
from gro821.sandbox.fmt import ANSI
from gro821.sandbox.render.plots import plot_benchmark_dashboard


def _results() -> ComparisonResults:
    return ComparisonResults(
        samples=[
            BenchmarkSample(
                method="kd-tree",
                sample_index=0,
                build_time_ns=10,
                query_time_ns=20,
                storage_bytes=100,
            ),
            BenchmarkSample(
                method="space-quadtree",
                sample_index=0,
                build_time_ns=30,
                query_time_ns=60,
                storage_bytes=200,
            ),
        ]
    )


def test_dashboard_uses_shared_time_limits_and_kdtree_baseline() -> None:
    figure = plot_benchmark_dashboard(_results(), time_baseline="kd-tree")

    build_axes, query_axes, storage_axes = figure.axes
    assert build_axes.get_ylim() == query_axes.get_ylim()
    assert [patch.get_height() for patch in build_axes.patches] == [1.0, 3.0]  # pyright: ignore[reportAttributeAccessIssue]
    assert [patch.get_height() for patch in query_axes.patches] == [1.0, 3.0]  # pyright: ignore[reportAttributeAccessIssue]
    assert [patch.get_height() for patch in storage_axes.patches] == [100.0, 200.0]  # pyright: ignore[reportAttributeAccessIssue]
    plt.close(figure)


def test_dashboard_rejects_unknown_time_baseline() -> None:
    try:
        plot_benchmark_dashboard(_results(), time_baseline="missing")
    except ValueError:
        pass
    else:
        raise AssertionError("An unknown time baseline must be rejected.")


def main():
    test_dashboard_uses_shared_time_limits_and_kdtree_baseline()
    test_dashboard_rejects_unknown_time_baseline()

    print(f"Plotting: {ANSI.BRIGHT_GREEN}All tests passed.{ANSI.RESET}")


if __name__ == "__main__":
    main()
