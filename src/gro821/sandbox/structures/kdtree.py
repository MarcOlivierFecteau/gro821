from dataclasses import dataclass
from typing import Literal

from gro821.sandbox.geometry import AABB, OBB, Point, aabb_intersects, aabb_intersects_obb_sat

type Axis = Literal[0, 1]


@dataclass
class KdTreeNode:
    bb: AABB
    split_axis: Axis | None
    split_value: float | None
    lhs: KdTreeNode | None
    rhs: KdTreeNode | None
    points: list[Point] | None

    def is_leaf(self) -> bool:
        return all([self.lhs is None, self.rhs is None, self.points is not None])

    def collect_points(self) -> list[Point]:
        if self.is_leaf():
            return self.points or []

        return self.lhs.collect_points() + self.rhs.collect_points()  # pyright: ignore[reportOptionalMemberAccess]


def build_kdtree(points: list[Point], region: AABB, capacity: int, _depth: int = 0) -> KdTreeNode:
    """Returns the root node. NOTE: `_depth` is passthrough only and is not meant to be specified."""
    points = [p for p in points if region.contains(p)]
    if not points:
        return KdTreeNode(region, None, None, None, None, [])

    if len(points) <= capacity:
        return KdTreeNode(region, None, None, None, None, points)

    axis: Axis = _depth % 2  # pyright: ignore[reportAssignmentType]

    sorted_points = sorted(points, key=lambda p: (p.x, p.y) if axis == 0 else (p.y, p.x))

    median_idx = len(sorted_points) // 2
    median_point = sorted_points[median_idx]
    split_value = median_point.coordinate(axis)

    if all(p.coordinate(axis) == split_value for p in points):
        return KdTreeNode(region, None, None, None, None, points)

    left_points = sorted_points[:median_idx]
    right_points = sorted_points[median_idx:]

    if axis == 0:
        left_region = AABB(region.start, Point(split_value, region.end.y))
        right_region = AABB(Point(split_value, region.start.y), region.end)
    else:
        left_region = AABB(region.start, Point(region.end.x, split_value))
        right_region = AABB(Point(region.start.x, split_value), region.end)

    left = build_kdtree(left_points, left_region, capacity, _depth + 1)
    right = build_kdtree(right_points, right_region, capacity, _depth + 1)

    return KdTreeNode(region, axis, split_value, left, right, None)


def kdtree_collides_bb(node: KdTreeNode, bb: AABB | OBB) -> bool:
    if isinstance(bb, AABB):
        if not aabb_intersects(node.bb, bb):
            return False
    else:
        if not aabb_intersects_obb_sat(node.bb, bb):
            return False

    if node.is_leaf():
        for p in node.points or []:
            if bb.contains(p):
                return True
        return False

    return kdtree_collides_bb(node.lhs, bb) or kdtree_collides_bb(node.rhs, bb)  # pyright: ignore[reportArgumentType]


def kdtree_collides_hybrid(node: KdTreeNode, obb: OBB) -> bool:
    """Use the OBB's enclosing AABB for pruning and the OBB for point hits."""
    return _kdtree_collides_hybrid(node, obb, obb.enclosing_aabb())


def _kdtree_collides_hybrid(node: KdTreeNode, obb: OBB, broad_phase: AABB) -> bool:
    if not aabb_intersects(node.bb, broad_phase):
        return False

    if node.is_leaf():
        return any(obb.contains(point) for point in node.points or [])

    return _kdtree_collides_hybrid(node.lhs, obb, broad_phase) or _kdtree_collides_hybrid(node.rhs, obb, broad_phase)  # pyright: ignore[reportArgumentType]
