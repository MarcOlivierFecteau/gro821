from dataclasses import dataclass
from typing import Literal

from gro821.sandbox.geometry import AABB, OBB, Point, aabb_intersects, aabb_intersects_obb_sat

type Quadrant = Literal["NW", "NE", "SW", "SE"]
type QuadrantPath = tuple[Quadrant, ...]


@dataclass
class SpaceQuadtreeNode:
    bb: AABB
    NW: SpaceQuadtreeNode | None
    NE: SpaceQuadtreeNode | None
    SW: SpaceQuadtreeNode | None
    SE: SpaceQuadtreeNode | None
    points: list[Point] | None

    def is_leaf(self) -> bool:
        return all(
            [
                self.NW is None,
                self.NE is None,
                self.SW is None,
                self.SE is None,
            ]
        )

    def size(self) -> int:
        return 0 if not self.is_leaf() else len(self.points)  # pyright: ignore[reportArgumentType]

    def count_points(self) -> int:
        if self.is_leaf():
            return len(self.points or [])

        return sum(
            child.count_points()
            for child in [self.NW, self.NE, self.SW, self.SE]
            if child is not None
        )


@dataclass
class PointQuadtreeNode:
    bb: AABB
    split_point: Point | None
    NW: PointQuadtreeNode | None
    NE: PointQuadtreeNode | None
    SW: PointQuadtreeNode | None
    SE: PointQuadtreeNode | None
    points: list[Point] | None

    def is_leaf(self) -> bool:
        return all(
            [
                self.split_point is None,
                self.NW is None,
                self.NE is None,
                self.SW is None,
                self.SE is None,
            ]
        )


@dataclass
class CompressedQuadtreeNode:
    bb: AABB
    points: list[Point] | None  # internal nodes: None; [] valid for empty leaf
    children: dict[QuadrantPath, CompressedQuadtreeNode]  # key describes quadrants skipped to child

    def is_leaf(self) -> bool:
        return self.points is not None


def build_space_quadtree(points: list[Point], region: AABB, capacity: int = 4) -> SpaceQuadtreeNode:
    """Returns the root node."""
    points = [p for p in points if region.contains(p)]
    if not points:
        return SpaceQuadtreeNode(region, None, None, None, None, [])
    if len(points) <= capacity:
        return SpaceQuadtreeNode(region, None, None, None, None, points)

    x_mid = (region.start.x + region.end.x) / 2
    y_mid = (region.start.y + region.end.y) / 2

    NW_points = [p for p in points if p.x <= x_mid and p.y > y_mid]
    NE_points = [p for p in points if p.x > x_mid and p.y > y_mid]
    SW_points = [p for p in points if p.x <= x_mid and p.y <= y_mid]
    SE_points = [p for p in points if p.x > x_mid and p.y <= y_mid]

    NW_region = AABB(Point(region.start.x, y_mid), Point(x_mid, region.end.y))
    NE_region = AABB(Point(x_mid, y_mid), region.end)
    SW_region = AABB(region.start, Point(x_mid, y_mid))
    SE_region = AABB(Point(x_mid, region.start.y), Point(region.end.x, y_mid))

    NW = build_space_quadtree(NW_points, NW_region, capacity)
    NE = build_space_quadtree(NE_points, NE_region, capacity)
    SW = build_space_quadtree(SW_points, SW_region, capacity)
    SE = build_space_quadtree(SE_points, SE_region, capacity)

    return SpaceQuadtreeNode(region, NW, NE, SW, SE, None)


def compress_quadtree(node: SpaceQuadtreeNode) -> CompressedQuadtreeNode:
    if node.is_leaf():
        return CompressedQuadtreeNode(node.bb, node.points, {})

    occupied: list[tuple[Quadrant, SpaceQuadtreeNode]] = []
    for quadrant, child in [("NW", node.NW), ("NE", node.NE), ("SW", node.SW), ("SE", node.SE)]:
        if child is not None and child.count_points() >= 1:
            occupied.append((quadrant, child))  # pyright: ignore[reportArgumentType]

    if not occupied:
        return CompressedQuadtreeNode(node.bb, [], {})

    if len(occupied) > 1:
        children: dict[QuadrantPath, CompressedQuadtreeNode] = {}

        for quadrant, child in occupied:
            compressed_child = compress_quadtree(child)
            children[(quadrant,)] = compressed_child

        return CompressedQuadtreeNode(node.bb, None, children)

    # Exactly one non-empty child -> skip unary chain
    quadrant, child = occupied[0]
    path: list[Quadrant] = [quadrant]

    while not child.is_leaf():
        child_occupied = []
        for next_quadrant, next_child in [
            ("NW", child.NW),
            ("NE", child.NE),
            ("SW", child.SW),
            ("SE", child.SE),
        ]:
            if next_child is not None and next_child.count_points() >= 1:
                child_occupied.append((next_quadrant, next_child))

        if len(child_occupied) != 1:
            break

        next_quadrant, child = child_occupied[0]
        path.append(next_quadrant)

    compressed_child = compress_quadtree(child)

    return CompressedQuadtreeNode(node.bb, None, {tuple(path): compressed_child})


def build_point_quadtree(points: list[Point], region: AABB, capacity: int = 4, _depth: int = 0):
    """NOTE: `_depth` is passthrough only and is not meant to be specified."""
    points = [p for p in points if region.contains(p)]  # Filter points outside the region
    if not points:
        return PointQuadtreeNode(region, None, None, None, None, None, [])
    if len(points) <= capacity:
        return PointQuadtreeNode(region, None, None, None, None, None, points)

    # NOTE: Alternating median axis keeps the quadtree more balanced.
    axis = _depth % 2  # 0 -> x-axis, 1 -> y-axis.
    sorted_axis = sorted(points, key=lambda p: p.x if axis == 0 else p.y)
    split_point = sorted_axis[len(points) // 2]

    # fmt: off
    NW_points = [p for p in points if p != split_point and p.x <= split_point.x and p.y > split_point.y]
    NE_points = [p for p in points if p != split_point and p.x > split_point.x and p.y > split_point.y]
    SW_points = [p for p in points if p != split_point and p.x <= split_point.x and p.y <= split_point.y]
    SE_points = [p for p in points if p != split_point and p.x > split_point.x and p.y <= split_point.y]
    # fmt: on

    NW_region = AABB(Point(region.start.x, split_point.y), Point(split_point.x, region.end.y))
    NE_region = AABB(Point(split_point.x, split_point.y), region.end)
    SW_region = AABB(region.start, Point(split_point.x, split_point.y))
    SE_region = AABB(Point(split_point.x, region.start.y), Point(region.end.x, split_point.y))

    NW = build_point_quadtree(NW_points, NW_region, capacity, _depth + 1)
    NE = build_point_quadtree(NE_points, NE_region, capacity, _depth + 1)
    SW = build_point_quadtree(SW_points, SW_region, capacity, _depth + 1)
    SE = build_point_quadtree(SE_points, SE_region, capacity, _depth + 1)

    return PointQuadtreeNode(region, split_point, NW, NE, SW, SE, None)


def space_quadtree_collides_aabb(quadtree: SpaceQuadtreeNode, aabb: AABB) -> bool:
    """Return whether any point in ``quadtree`` lies inside ``aabb``.

    A point on the boundary of ``aabb`` is considered a collision. Nodes whose
    bounding boxes do not intersect ``aabb`` are skipped entirely.
    """
    if not aabb_intersects(quadtree.bb, aabb):
        return False

    if quadtree.is_leaf():
        return any(aabb.contains(point) for point in quadtree.points or [])

    return any(
        child is not None and space_quadtree_collides_aabb(child, aabb)
        for child in [quadtree.NW, quadtree.NE, quadtree.SW, quadtree.SE]
    )


def space_quadtree_collides_obb(quadtree: SpaceQuadtreeNode, obb: OBB) -> bool:
    """Return whether any point in ``quadtree`` lies inside ``obb``.

    Nodes are pruned with the separating axis theorem against the OBB, then
    points are tested against the OBB itself in its local coordinate frame.
    """
    if not aabb_intersects_obb_sat(quadtree.bb, obb):
        return False

    if quadtree.is_leaf():
        return any(obb.contains(point) for point in quadtree.points or [])

    return any(
        child is not None and space_quadtree_collides_obb(child, obb)
        for child in [quadtree.NW, quadtree.NE, quadtree.SW, quadtree.SE]
    )


def space_quadtree_collides_hybrid(quadtree: SpaceQuadtreeNode, obb: OBB) -> bool:
    """Use the OBB's enclosing AABB for pruning and the OBB for point hits."""
    return _space_quadtree_collides_hybrid(quadtree, obb, obb.enclosing_aabb())


def _space_quadtree_collides_hybrid(
    quadtree: SpaceQuadtreeNode, obb: OBB, broad_phase: AABB
) -> bool:
    if not aabb_intersects(quadtree.bb, broad_phase):
        return False

    if quadtree.is_leaf():
        return any(obb.contains(point) for point in quadtree.points or [])

    return any(
        child is not None and _space_quadtree_collides_hybrid(child, obb, broad_phase)
        for child in [quadtree.NW, quadtree.NE, quadtree.SW, quadtree.SE]
    )


def point_quadtree_collides_aabb(node: PointQuadtreeNode, aabb: AABB) -> bool:
    if not aabb_intersects(node.bb, aabb):
        return False

    if node.is_leaf():
        return any(aabb.contains(p) for p in node.points or [])

    if aabb.contains(node.split_point):  # pyright: ignore[reportArgumentType]
        return True

    return any(
        child is not None and point_quadtree_collides_aabb(child, aabb)
        for child in [node.NW, node.NE, node.SW, node.SE]
    )


def point_quadtree_collides_obb(node: PointQuadtreeNode, obb: OBB) -> bool:
    """Return whether any point in ``quadtree`` lies inside ``obb``.

    Nodes are pruned with the separating axis theorem against the OBB, then
    points are tested against the OBB itself in its local coordinate frame.
    """
    if not aabb_intersects_obb_sat(node.bb, obb):
        return False

    if node.is_leaf():
        return any(obb.contains(point) for point in node.points or [])

    if obb.contains(node.split_point):  # pyright: ignore[reportArgumentType]
        return True

    return any(
        child is not None and point_quadtree_collides_obb(child, obb)
        for child in [node.NW, node.NE, node.SW, node.SE]
    )


def point_quadtree_collides_hybrid(node: PointQuadtreeNode, obb: OBB) -> bool:
    """Use the OBB's enclosing AABB for pruning and the OBB for point hits."""
    return _point_quadtree_collides_hybrid(node, obb, obb.enclosing_aabb())


def _point_quadtree_collides_hybrid(
    node: PointQuadtreeNode, obb: OBB, broad_phase: AABB
) -> bool:
    if not aabb_intersects(node.bb, broad_phase):
        return False

    if node.is_leaf():
        return any(obb.contains(point) for point in node.points or [])

    if obb.contains(node.split_point):  # pyright: ignore[reportArgumentType]
        return True

    return any(
        child is not None and _point_quadtree_collides_hybrid(child, obb, broad_phase)
        for child in [node.NW, node.NE, node.SW, node.SE]
    )


def compressed_quadtree_collides_aabb(node: CompressedQuadtreeNode, aabb: AABB) -> bool:
    if not aabb_intersects(node.bb, aabb):
        return False

    if node.is_leaf():
        for p in node.points or []:
            if aabb.contains(p):
                return True
        return False

    for child in node.children.values():
        if compressed_quadtree_collides_aabb(child, aabb):
            return True

    return False


def compressed_quadtree_collides_obb(node: CompressedQuadtreeNode, obb: OBB) -> bool:
    if not aabb_intersects_obb_sat(node.bb, obb):
        return False

    if node.is_leaf():
        for p in node.points or []:
            if obb.contains(p):
                return True
        return False

    for child in node.children.values():
        if compressed_quadtree_collides_obb(child, obb):
            return True

    return False


def compressed_quadtree_collides_hybrid(node: CompressedQuadtreeNode, obb: OBB) -> bool:
    """Use the OBB's enclosing AABB for pruning and the OBB for point hits."""
    return _compressed_quadtree_collides_hybrid(node, obb, obb.enclosing_aabb())


def _compressed_quadtree_collides_hybrid(
    node: CompressedQuadtreeNode, obb: OBB, broad_phase: AABB
) -> bool:
    if not aabb_intersects(node.bb, broad_phase):
        return False

    if node.is_leaf():
        return any(obb.contains(point) for point in node.points or [])

    return any(
        _compressed_quadtree_collides_hybrid(child, obb, broad_phase)
        for child in node.children.values()
    )
