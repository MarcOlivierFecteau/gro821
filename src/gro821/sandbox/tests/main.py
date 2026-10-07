from gro821.sandbox.tests import (
    test_benchmark,
    test_comparison,
    test_kdtree,
    test_memory,
    test_plots,
    test_quadtree,
)


def main():
    test_quadtree.main()
    test_kdtree.main()
    test_memory.main()
    test_benchmark.main()
    test_comparison.main()
    test_plots.main()


if __name__ == "__main__":
    main()
