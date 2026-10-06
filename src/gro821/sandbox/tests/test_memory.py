from gro821.sandbox.fmt import ANSI
from gro821.sandbox.geometry import AABB, Point
from gro821.sandbox.memory import retained_size, retained_size_breakdown
from gro821.sandbox.structures.kdtree import build_kdtree
from gro821.sandbox.structures.quadtree import (
    build_point_quadtree,
    build_space_quadtree,
    compress_quadtree,
)


def make_points(count: int) -> list[Point]:
    return [Point(index * 10 + 5, index * 7 + 5) for index in range(count)]


def make_region() -> AABB:
    return AABB(Point(0, 0), Point(100, 100))


class TestRetainedSize:
    def test_empty_root_has_a_positive_footprint(self):
        assert retained_size([]) > 0

    def test_shared_objects_are_counted_once(self):
        point = Point(10, 10)
        shared = retained_size([point, point])
        distinct = retained_size([Point(10, 10), Point(10, 10)])

        assert shared < distinct

    def test_breakdown_sums_to_total(self):
        points = make_points(4)
        breakdown = retained_size_breakdown(points)

        assert breakdown
        assert sum(breakdown.values()) == retained_size(points)
        assert breakdown["Point"] > 0

    def test_larger_point_lists_have_larger_footprints(self):
        assert retained_size(make_points(2)) < retained_size(make_points(8))


class TestTreeFootprints:
    def test_kdtree_footprint_includes_tree_overhead(self):
        points = make_points(8)
        tree = build_kdtree(points, make_region(), capacity=1)

        assert retained_size(tree) > retained_size(points)

    def test_space_quadtree_footprint_is_measureable(self):
        tree = build_space_quadtree(make_points(8), make_region(), capacity=1)

        assert retained_size(tree) > 0

    def test_point_quadtree_footprint_is_measureable(self):
        tree = build_point_quadtree(make_points(8), make_region(), capacity=1)

        assert retained_size(tree) > 0

    def test_compressed_quadtree_footprint_is_measureable(self):
        tree = compress_quadtree(build_space_quadtree(make_points(8), make_region(), capacity=1))

        assert retained_size(tree) > 0

    def test_tree_breakdown_contains_node_types(self):
        tree = build_kdtree(make_points(4), make_region(), capacity=1)
        breakdown = retained_size_breakdown(tree)

        assert breakdown["KdTreeNode"] > 0
        assert breakdown["AABB"] > 0


def main():
    test_retained_size = TestRetainedSize()
    test_retained_size.test_empty_root_has_a_positive_footprint()
    test_retained_size.test_shared_objects_are_counted_once()
    test_retained_size.test_breakdown_sums_to_total()
    test_retained_size.test_larger_point_lists_have_larger_footprints()

    test_tree_footprints = TestTreeFootprints()
    test_tree_footprints.test_kdtree_footprint_includes_tree_overhead()
    test_tree_footprints.test_space_quadtree_footprint_is_measureable()
    test_tree_footprints.test_point_quadtree_footprint_is_measureable()
    test_tree_footprints.test_compressed_quadtree_footprint_is_measureable()
    test_tree_footprints.test_tree_breakdown_contains_node_types()

    print(f"Memory: {ANSI.BRIGHT_GREEN}All tests passed.{ANSI.RESET}")


if __name__ == "__main__":
    main()
