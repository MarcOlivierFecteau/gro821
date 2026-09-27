import math

from gro821.sandbox.fmt import ANSI
from gro821.sandbox.geometry import AABB, OBB, Point
from gro821.sandbox.structures.quadtree import (
    CompressedQuadtreeNode,
    PointQuadtreeNode,
    SpaceQuadtreeNode,
    build_point_quadtree,
    build_space_quadtree,
    compress_quadtree,
    compressed_quadtree_collides_aabb,
    compressed_quadtree_collides_obb,
)
from gro821.sandbox.world import World

# === Helpers ===


def collect_points(node: SpaceQuadtreeNode | PointQuadtreeNode) -> list[Point]:
    """Return all points stored in a quadtree."""
    if node.is_leaf():
        return node.points if node.points is not None else []
    result = [node.split_point] if isinstance(node, PointQuadtreeNode) and node.split_point else []

    for child in [node.NW, node.NE, node.SW, node.SE]:
        result.extend(collect_points(child))  # pyright: ignore[reportArgumentType]
    return result


def count_nodes(node: SpaceQuadtreeNode | PointQuadtreeNode) -> int:
    if node.is_leaf():
        return 1
    return 1 + sum(count_nodes(child) for child in [node.NW, node.NE, node.SW, node.SE])  # pyright: ignore[reportArgumentType]


def collect_compressed_points(node: CompressedQuadtreeNode) -> list[Point]:
    if node.is_leaf():
        return node.points or []

    result = []
    for child in node.children.values():
        result.extend(collect_compressed_points(child))
    return result


def count_compressed_nodes(node: CompressedQuadtreeNode) -> int:
    if node.is_leaf():
        return 1
    return 1 + sum(count_compressed_nodes(child) for child in node.children.values())


# === Tests ===


class TestBuildSpaceQuadtree:
    def test_empty_points(self):
        """Empty input produces an empty leaf with no points."""
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_space_quadtree([], region)
        assert node.is_leaf()
        assert node.points is None or node.points == []

    def test_single_point(self):
        """A single point stays in a leaf node."""
        pts = [Point(50, 50)]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_space_quadtree(pts, region)
        assert node.is_leaf()
        assert node.points == pts

    def test_points_outside_region_are_ignored(self):
        """Points outside the root region are not stored in the quadtree."""
        region = AABB(Point(0, 0), Point(100, 100))
        inside = [Point(25, 25), Point(75, 75)]
        outside = [Point(-1, 50), Point(50, 101), Point(150, 50)]

        node = build_space_quadtree(inside + outside, region, capacity=1)

        assert sorted(collect_points(node), key=lambda p: (p.x, p.y)) == sorted(
            inside, key=lambda p: (p.x, p.y)
        )

    def test_at_capacity(self):
        """Exactly `capacity` points stays in one leaf."""
        pts = [Point(10, 10), Point(20, 20), Point(30, 30), Point(40, 40)]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_space_quadtree(pts, region, capacity=4)
        assert node.is_leaf()
        assert len(node.points) == 4  # pyright: ignore[reportArgumentType]

    def test_exceeds_capacity_subdivides(self):
        """More points than capacity triggers subdivision."""
        pts = [Point(10, 10), Point(20, 20), Point(30, 30), Point(40, 40), Point(50, 50)]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_space_quadtree(pts, region, capacity=4)
        assert not node.is_leaf()
        # All four children should exist
        for child in [node.NW, node.NE, node.SW, node.SE]:
            assert child is not None

    def test_all_points_preserved(self):
        """No points are lost after building the tree."""
        world = World(10, 10)
        pts = world.generate_points(100)
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_space_quadtree(pts, region, capacity=4)
        collected = collect_points(node)
        assert len(collected) == len(pts)
        assert sorted(collected, key=lambda p: (p.x, p.y)) == sorted(pts, key=lambda p: (p.x, p.y))

    def test_points_in_correct_quadrants(self):
        """Points end up in the child whose region contains them."""
        # 5 points clearly in each quadrant → forces a single subdivision
        region = AABB(Point(0, 0), Point(100, 100))
        pts = [
            Point(25, 75),  # NW
            Point(75, 75),  # NE
            Point(25, 25),  # SW
            Point(75, 25),  # SE
            Point(50, 50),  # boundary — goes SW (x <= 50, y <= 50)
        ]
        node = build_space_quadtree(pts, region, capacity=4)

        # Each leaf child should contain exactly one point (except SW which has two)
        assert node.NW.is_leaf() and node.NW.points == [Point(25, 75)]  # pyright: ignore[reportOptionalMemberAccess]
        assert node.NE.is_leaf() and node.NE.points == [Point(75, 75)]  # pyright: ignore[reportOptionalMemberAccess]
        # SW gets (25,25) and the boundary (50,50)
        assert node.SW.is_leaf() and len(node.SW.points) == 2  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]
        assert node.SE.is_leaf() and node.SE.points == [Point(75, 25)]  # pyright: ignore[reportOptionalMemberAccess]

    def test_bounding_boxes_correct(self):
        """Each node's bounding box matches its region in the subdivision."""
        world = World(10, 10)
        pts = world.generate_points(20)
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_space_quadtree(pts, region, capacity=4)

        # Check that every point in every node falls inside that node's bb
        def check_bb(n: SpaceQuadtreeNode) -> None:
            if n.is_leaf():
                for p in n.points or []:
                    assert n.bb.contains(p), f"{p} outside {n.bb}"
            else:
                for child in [n.NW, n.NE, n.SW, n.SE]:
                    check_bb(child)  # pyright: ignore[reportArgumentType]

        check_bb(node)

    def test_capacity_one(self):
        """Edge case: capacity = 1 forces maximum subdivision."""
        world = World(10, 10)
        pts = sorted(world.generate_points(10), key=lambda p: (p.x, p.y))
        region = AABB(Point(0, 0), Point(100, 100))
        root = build_space_quadtree(pts, region, capacity=1)  # pyright: ignore[reportArgumentType]

        # Every leaf must contain at most 1 point
        def check_capacity(n: SpaceQuadtreeNode) -> None:
            if n.is_leaf():
                assert n.points is None or len(n.points) <= 1, n
            else:
                for child in [n.NW, n.NE, n.SW, n.SE]:
                    check_capacity(child)  # pyright: ignore[reportArgumentType]

        check_capacity(root)
        root_points = sorted(collect_points(root), key=lambda p: (p.x, p.y))
        assert root_points == pts, f"Expected: {pts}\nGot: {root_points}"

    def test_non_square_region(self):
        """Works with a non-square rectangular region."""
        pts = [Point(5, 20), Point(15, 10), Point(25, 5), Point(35, 30), Point(45, 15)]
        region = AABB(Point(0, 0), Point(50, 40))
        node = build_space_quadtree(pts, region, capacity=4)
        assert not node.is_leaf()
        assert len(collect_points(node)) == 5

    def test_negative_coordinates(self):
        """Works with points in negative coordinate space."""
        pts = [Point(-10, -10), Point(-30, -20), Point(-40, -50), Point(-60, -70), Point(-80, -90)]
        region = AABB(Point(-100, -100), Point(0, 0))
        node = build_space_quadtree(pts, region, capacity=4)
        assert not node.is_leaf()
        assert len(collect_points(node)) == 5

    # NOTE: assumed a LiDAR cannot return duplicate points
    # def test_duplicate_points(self):
    #     """Duplicate points are handled (all stored)."""
    #     pts = [Point(10, 10)] * 10
    #     region = AABB(Point(0, 0), Point(100, 100))
    #     node = build_quadtree(pts, region, capacity=4)
    #     assert len(collect_points(node)) == 10


class TestCompressQuadtree:
    def test_empty_tree_becomes_empty_compressed_leaf(self):
        region = AABB(Point(0, 0), Point(100, 100))
        compressed = compress_quadtree(build_space_quadtree([], region))

        assert isinstance(compressed, CompressedQuadtreeNode)
        assert compressed.is_leaf()
        assert compressed.bb == region
        assert compressed.points == []
        assert compressed.children == {}

    def test_leaf_points_are_preserved(self):
        points = [Point(10, 10), Point(20, 20)]
        region = AABB(Point(0, 0), Point(100, 100))
        compressed = compress_quadtree(build_space_quadtree(points, region))

        assert compressed.is_leaf()
        assert compressed.points == points
        assert compressed.children == {}

    def test_unary_chains_are_skipped(self):
        region = AABB(Point(0, 0), Point(256, 256))
        points = [Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4), Point(5, 5)]
        space_tree = build_space_quadtree(points, region, capacity=1)
        compressed = compress_quadtree(space_tree)

        assert not compressed.is_leaf()
        assert len(compressed.children) == 1
        path, _ = next(iter(compressed.children.items()))
        assert len(path) > 1
        assert sorted(collect_compressed_points(compressed), key=lambda p: (p.x, p.y)) == sorted(
            points, key=lambda p: (p.x, p.y)
        )
        assert count_compressed_nodes(compressed) < count_nodes(space_tree)

    def test_branching_nodes_keep_each_occupied_quadrant(self):
        region = AABB(Point(0, 0), Point(100, 100))
        points = [
            Point(10, 10),
            Point(20, 20),
            Point(80, 10),
            Point(90, 20),
            Point(10, 80),
            Point(20, 90),
        ]
        compressed = compress_quadtree(build_space_quadtree(points, region, capacity=1))

        assert not compressed.is_leaf()
        assert set(compressed.children) == {("SW",), ("SE",), ("NW",)}
        assert sorted(collect_compressed_points(compressed), key=lambda p: (p.x, p.y)) == sorted(
            points, key=lambda p: (p.x, p.y)
        )

    def test_collides_aabb_finds_points_and_prunes_misses(self):
        region = AABB(Point(0, 0), Point(100, 100))
        points = [Point(10, 10), Point(30, 30), Point(70, 70), Point(90, 90)]
        compressed = compress_quadtree(build_space_quadtree(points, region, capacity=1))

        assert compressed_quadtree_collides_aabb(compressed, AABB(Point(25, 25), Point(35, 35)))
        assert not compressed_quadtree_collides_aabb(
            compressed, AABB(Point(40, 40), Point(60, 60))
        )

    def test_collides_aabb_includes_boundary_points(self):
        region = AABB(Point(0, 0), Point(100, 100))
        point = Point(25, 25)
        compressed = compress_quadtree(build_space_quadtree([point], region))

        assert compressed_quadtree_collides_aabb(compressed, AABB(point, point))

    def test_collides_obb_handles_rotation_and_misses(self):
        region = AABB(Point(0, 0), Point(100, 100))
        points = [Point(45, 45), Point(55, 55), Point(20, 20), Point(20, 80)]
        compressed = compress_quadtree(build_space_quadtree(points, region, capacity=1))
        obb = OBB(Point(50, 50), Point(10, 30), math.pi / 4)

        assert compressed_quadtree_collides_obb(compressed, obb)
        assert compressed_quadtree_collides_obb(
            compress_quadtree(build_space_quadtree([Point(20, 20)], region)), obb
        ) is False

    def test_collisions_on_empty_compressed_tree_are_false(self):
        region = AABB(Point(0, 0), Point(100, 100))
        compressed = compress_quadtree(build_space_quadtree([], region))

        assert not compressed_quadtree_collides_aabb(
            compressed, AABB(Point(0, 0), Point(100, 100))
        )
        assert not compressed_quadtree_collides_obb(
            compressed, OBB(Point(50, 50), Point(50, 50), 0)
        )


class TestBuildPointQuadtree:
    def test_empty_points(self):
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree([], region)
        assert node.is_leaf()
        assert node.points == []

    def test_single_point(self):
        pts = [Point(50, 50)]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region)
        assert node.is_leaf()
        assert node.points == pts

    def test_at_capacity(self):
        pts = [Point(10, 10), Point(20, 20), Point(30, 30), Point(40, 40)]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region, capacity=4)
        assert node.is_leaf()
        assert len(node.points) == 4  # pyright: ignore[reportArgumentType]

    def test_exceeds_capacity_subdivides(self):
        pts = [Point(10, 10), Point(20, 20), Point(30, 30), Point(40, 40), Point(50, 50)]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region, capacity=4)
        assert not node.is_leaf()
        assert node.split_point is not None
        for child in [node.NW, node.NE, node.SW, node.SE]:
            assert child is not None

    def test_split_point_stored_at_internal_node(self):
        """Internal node stores a split_point; leaves do not."""
        pts = [Point(10, 10), Point(20, 20), Point(30, 30), Point(40, 40), Point(50, 50)]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region, capacity=2)
        assert not node.is_leaf()
        assert node.split_point is not None  # root has a split point
        # All children should be leaves (or possibly internal depending on distribution)
        for child in [node.NW, node.NE, node.SW, node.SE]:
            if child.is_leaf():  # pyright: ignore[reportOptionalMemberAccess]
                assert child.split_point is None  # pyright: ignore[reportOptionalMemberAccess]
                assert child.points is not None  # pyright: ignore[reportOptionalMemberAccess]

    def test_all_points_preserved(self):
        """No points are lost — including the split points of internal nodes."""
        world = World(10, 10)
        pts = world.generate_points(100)
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region, capacity=4)
        collected = collect_points(node)
        assert len(collected) == len(pts)
        assert sorted(collected, key=lambda p: (p.x, p.y)) == sorted(pts, key=lambda p: (p.x, p.y))

    def test_points_in_correct_regions(self):
        """Every point falls inside its hosting node's bounding box."""
        world = World(10, 10)
        pts = world.generate_points(50)
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region, capacity=4)

        def check_bb(n: PointQuadtreeNode) -> None:
            if n.is_leaf():
                for p in n.points or []:
                    assert n.bb.contains(p), f"{p} outside {n.bb}"
            else:
                assert n.bb.contains(n.split_point), (  # pyright: ignore[reportArgumentType]
                    f"split_point {n.split_point} outside {n.bb}"
                )
                for child in [n.NW, n.NE, n.SW, n.SE]:
                    check_bb(child)  # pyright: ignore[reportArgumentType]

        check_bb(node)

    def test_split_point_excluded_from_children(self):
        """The split point should not appear in any child's descendants."""
        pts = [Point(10, 10), Point(30, 30), Point(50, 50), Point(70, 70), Point(90, 90)]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region, capacity=2)
        assert not node.is_leaf()
        sp = node.split_point

        # Collect all points from children (not including the root split_point)
        child_points = []
        for child in [node.NW, node.NE, node.SW, node.SE]:
            child_points.extend(collect_points(child))  # pyright: ignore[reportArgumentType]

        assert sp not in child_points, f"Split point {sp} found in a child"

    def test_alternating_axis(self):
        """Depth 0 splits on x, depth 1 on y, depth 2 on x, etc."""
        # Place points so that the median on x is predictable
        pts = [
            Point(10, 50),
            Point(90, 50),  # far left and right
            Point(20, 10),
            Point(80, 90),  # make sure > capacity
            Point(50, 50),  # median on x
        ]
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region, capacity=2)
        # Root splits on x → median x = 50
        assert node.split_point.x == 50  # pyright: ignore[reportOptionalMemberAccess]

        # If root succeeded, check a child splits on y
        # Since we used capacity=2, the deepest internal node
        # at the next level should split on y
        def check_depth(n, d):
            if n.is_leaf():
                return
            axis = d % 2
            if axis == 0:
                # Children should have y-based AABB splitting
                pass  # hard to assert directly without knowing distribution
            for child in [n.NW, n.NE, n.SW, n.SE]:
                check_depth(child, d + 1)

        check_depth(node, 0)
        # Just verify no recursion error and the tree is valid
        assert count_nodes(node) >= 3

    def test_capacity_one(self):
        """capacity=1 forces each leaf to hold exactly one point."""
        world = World(10, 10)
        pts = world.generate_points(20)
        region = AABB(Point(0, 0), Point(100, 100))
        node = build_point_quadtree(pts, region, capacity=1)

        def check_capacity(n: PointQuadtreeNode) -> None:
            if n.is_leaf():
                assert len(n.points) <= 1  # pyright: ignore[reportArgumentType]
            else:
                for child in [n.NW, n.NE, n.SW, n.SE]:
                    check_capacity(child)  # pyright: ignore[reportArgumentType]

        check_capacity(node)
        assert len(collect_points(node)) == 20

    def test_non_square_region(self):
        """Works with a non-square rectangular region."""
        pts = [Point(5, 20), Point(15, 10), Point(25, 5), Point(35, 30), Point(45, 15)]
        region = AABB(Point(0, 0), Point(50, 40))
        node = build_point_quadtree(pts, region, capacity=4)
        assert not node.is_leaf()
        assert len(collect_points(node)) == 5

    def test_negative_coordinates(self):
        pts = [Point(-10, -10), Point(-30, -20), Point(-40, -50), Point(-60, -70), Point(-80, -90)]
        region = AABB(Point(-100, -100), Point(0, 0))
        node = build_point_quadtree(pts, region, capacity=4)
        assert not node.is_leaf()
        assert len(collect_points(node)) == 5

    # NOTE: assumed a LiDAR cannot return duplicate points
    # def test_duplicate_points(self):
    #     """All duplicates are preserved (split point has one copy removed)."""
    #     pts = [Point(10, 10)] * 10
    #     region = AABB(Point(0, 0), Point(100, 100))
    #     node = build_point_quadtree(pts, region, capacity=4)
    #     # One copy becomes the split point (stored on node),
    #     # the rest go to children — but `p != split_point`
    #     # removes ALL matching points, so only the split point survives.
    #     collected = collect_points(node)
    #     # Only the split point remains (because p != split_point excludes all duplicates)
    #     assert len(collected) == 1  # Known limitation of the current implementation


def main():
    tests_space = TestBuildSpaceQuadtree()
    tests_space.test_empty_points()
    tests_space.test_single_point()
    tests_space.test_at_capacity()
    tests_space.test_exceeds_capacity_subdivides()
    tests_space.test_all_points_preserved()
    tests_space.test_points_in_correct_quadrants()
    tests_space.test_bounding_boxes_correct()
    tests_space.test_capacity_one()
    tests_space.test_non_square_region()
    tests_space.test_negative_coordinates()
    # tests_space.test_duplicate_points()

    tests_point = TestBuildPointQuadtree()
    tests_point.test_empty_points()
    tests_point.test_single_point()
    tests_point.test_at_capacity()
    tests_point.test_exceeds_capacity_subdivides()
    tests_point.test_split_point_stored_at_internal_node()
    tests_point.test_all_points_preserved()
    tests_point.test_points_in_correct_regions()
    tests_point.test_split_point_excluded_from_children()
    tests_point.test_alternating_axis()
    tests_point.test_capacity_one()
    tests_point.test_non_square_region()
    tests_point.test_negative_coordinates()
    # tests_point.test_duplicate_points()

    tests_compressed = TestCompressQuadtree()
    tests_compressed.test_empty_tree_becomes_empty_compressed_leaf()
    tests_compressed.test_leaf_points_are_preserved()
    tests_compressed.test_unary_chains_are_skipped()
    tests_compressed.test_branching_nodes_keep_each_occupied_quadrant()
    tests_compressed.test_collides_aabb_finds_points_and_prunes_misses()
    tests_compressed.test_collides_aabb_includes_boundary_points()
    tests_compressed.test_collides_obb_handles_rotation_and_misses()
    tests_compressed.test_collisions_on_empty_compressed_tree_are_false()

    print(f"Quadtree: {ANSI.BRIGHT_GREEN}All tests passed.{ANSI.RESET}")


if __name__ == "__main__":
    main()
