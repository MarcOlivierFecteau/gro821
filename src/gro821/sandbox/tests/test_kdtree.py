import math

from gro821.sandbox.fmt import ANSI
from gro821.sandbox.geometry import AABB, OBB, Point
from gro821.sandbox.structures.kdtree import (
    KdTreeNode,
    build_kdtree,
    kdtree_collides_bb,
    kdtree_collides_hybrid,
)


def count_nodes(node: KdTreeNode) -> int:
    if node.is_leaf():
        return 1

    return 1 + count_nodes(node.lhs) + count_nodes(node.rhs)  # pyright: ignore[reportArgumentType]


class TestBuildKdTree:
    def test_empty_points(self):
        region = AABB(Point(0, 0), Point(100, 100))

        node = build_kdtree([], region, capacity=1)

        assert node.is_leaf()
        assert node.bb == region
        assert node.points == []

    def test_single_point(self):
        point = Point(50, 50)
        region = AABB(Point(0, 0), Point(100, 100))

        node = build_kdtree([point], region, capacity=1)

        assert node.is_leaf()
        assert node.points == [point]

    def test_points_outside_region_are_ignored(self):
        region = AABB(Point(0, 0), Point(100, 100))
        inside = [Point(25, 25), Point(75, 75)]
        outside = [Point(-1, 50), Point(50, 101), Point(150, 50)]

        node = build_kdtree(inside + outside, region, capacity=4)

        assert node.collect_points() == inside

    def test_at_capacity_stays_leaf(self):
        points = [Point(10, 10), Point(20, 20), Point(30, 30), Point(40, 40)]
        region = AABB(Point(0, 0), Point(100, 100))

        node = build_kdtree(points, region, capacity=len(points))

        assert node.is_leaf()
        assert node.points == points

    def test_exceeding_capacity_subdivides(self):
        points = [
            Point(10, 10),
            Point(20, 80),
            Point(50, 50),
            Point(80, 20),
            Point(90, 90),
        ]
        region = AABB(Point(0, 0), Point(100, 100))

        node = build_kdtree(points, region, capacity=2)

        assert not node.is_leaf()
        assert node.split_axis == 0
        assert node.split_value == 50
        assert node.lhs is not None
        assert node.rhs is not None
        assert count_nodes(node) >= 3

    def test_all_points_are_preserved(self):
        points = [
            Point(10, 10),
            Point(20, 80),
            Point(50, 50),
            Point(80, 20),
            Point(90, 90),
        ]
        region = AABB(Point(0, 0), Point(100, 100))

        node = build_kdtree(points, region, capacity=1)

        assert sorted(node.collect_points(), key=lambda p: (p.x, p.y)) == sorted(
            points, key=lambda p: (p.x, p.y)
        )

    def test_points_remain_inside_their_node_regions(self):
        points = [
            Point(10, 10),
            Point(20, 80),
            Point(50, 50),
            Point(80, 20),
            Point(90, 90),
        ]
        region = AABB(Point(0, 0), Point(100, 100))

        def assert_regions(node: KdTreeNode) -> None:
            if node.is_leaf():
                assert all(node.bb.contains(point) for point in node.points or [])
                return

            assert node.lhs is not None
            assert node.rhs is not None
            assert_regions(node.lhs)
            assert_regions(node.rhs)

        assert_regions(build_kdtree(points, region, capacity=1))

    def test_alternates_split_axis(self):
        points = [
            Point(10, 10),
            Point(20, 20),
            Point(30, 80),
            Point(40, 90),
            Point(50, 30),
            Point(60, 40),
            Point(70, 50),
        ]
        region = AABB(Point(0, 0), Point(100, 100))

        root = build_kdtree(points, region, capacity=2)

        assert root.split_axis == 0
        assert root.lhs is not None
        assert root.lhs.split_axis == 1

    def test_identical_split_coordinates_stop_subdivision(self):
        points = [Point(25, 10), Point(25, 30), Point(25, 50)]
        region = AABB(Point(0, 0), Point(100, 100))

        node = build_kdtree(points, region, capacity=1)

        assert node.is_leaf()
        assert node.points == points


class TestKdTreeCollision:
    def test_aabb_finds_points_and_prunes_misses(self):
        region = AABB(Point(0, 0), Point(100, 100))
        points = [Point(10, 10), Point(30, 30), Point(70, 70), Point(90, 90)]
        node = build_kdtree(points, region, capacity=1)

        assert kdtree_collides_bb(node, AABB(Point(25, 25), Point(35, 35)))
        assert not kdtree_collides_bb(node, AABB(Point(40, 40), Point(60, 60)))

    def test_aabb_collision_includes_boundary_points(self):
        point = Point(25, 25)
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_kdtree([point], region, capacity=1)

        assert kdtree_collides_bb(node, AABB(point, point))

    def test_obb_handles_rotation_and_misses(self):
        region = AABB(Point(0, 0), Point(100, 100))
        points = [Point(45, 45), Point(55, 55), Point(20, 20), Point(20, 80)]
        node = build_kdtree(points, region, capacity=1)
        obb = OBB(Point(50, 50), Point(10, 30), math.pi / 4)

        assert kdtree_collides_bb(node, obb)
        assert not kdtree_collides_bb(build_kdtree([Point(20, 20)], region, capacity=1), obb)

    def test_empty_tree_never_collides(self):
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_kdtree([], region, capacity=1)

        assert not kdtree_collides_bb(node, AABB(Point(0, 0), Point(100, 100)))
        assert not kdtree_collides_bb(node, OBB(Point(50, 50), Point(50, 50), 0))

    def test_hybrid_uses_obb_after_aabb_candidate(self):
        region = AABB(Point(0, 0), Point(100, 100))
        obb = OBB(Point(50, 50), Point(10, 30), math.pi / 4)

        assert not kdtree_collides_hybrid(
            build_kdtree([Point(70, 70)], region, capacity=1), obb
        )
        assert kdtree_collides_hybrid(
            build_kdtree([Point(50, 50)], region, capacity=1), obb
        )


def main():
    tests_build_kdtree = TestBuildKdTree()
    tests_build_kdtree.test_empty_points()
    tests_build_kdtree.test_single_point()
    tests_build_kdtree.test_points_outside_region_are_ignored()
    tests_build_kdtree.test_at_capacity_stays_leaf()
    tests_build_kdtree.test_exceeding_capacity_subdivides()
    tests_build_kdtree.test_all_points_are_preserved()
    tests_build_kdtree.test_points_remain_inside_their_node_regions()
    tests_build_kdtree.test_alternates_split_axis()
    tests_build_kdtree.test_identical_split_coordinates_stop_subdivision()

    tests_kdtree_collision = TestKdTreeCollision()
    tests_kdtree_collision.test_aabb_finds_points_and_prunes_misses()
    tests_kdtree_collision.test_aabb_collision_includes_boundary_points()
    tests_kdtree_collision.test_obb_handles_rotation_and_misses()
    tests_kdtree_collision.test_empty_tree_never_collides()

    print(f"Kd-tree: {ANSI.BRIGHT_GREEN}All tests passed.{ANSI.RESET}")


if __name__ == "__main__":
    main()
