import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Final

import numpy as np

EPSILON: Final = 1e-9


@dataclass(slots=True)
class Point:
    x: float
    y: float

    def to_np(self) -> np.typing.NDArray[np.float64]:
        return np.array([self.x, self.y], dtype=np.float64)

    def coordinate(self, axis: int) -> float:
        if axis not in [0, 1]:
            raise ValueError("`axis` must be in [0, 1].")
        return self.x if axis == 0 else self.y

    def __abs__(self) -> Point:
        return Point(abs(self.x), abs(self.y))

    def __neg__(self) -> Point:
        return Point(-self.x, -self.y)

    def __add__(self, other: Point):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Point | float):
        """Hadamard product (element-wise)."""
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y)
        elif isinstance(other, (int, float)):
            return Point(self.x * other, self.y * other)
        else:
            raise TypeError(f"{type(other)} cannot be multiplied with `Point`")

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __ne__(self, other: object) -> bool:
        if other is None:
            return False
        if not isinstance(other, Point):
            raise TypeError("Can only compare with 'Point'.")
        return self.x != other.x or self.y != other.y

    def __lt__(self, other: Point) -> bool:
        return self.x < other.x and self.y < other.y

    def __le__(self, other: Point) -> bool:
        return self.x <= other.x and self.y <= other.y

    def __gt__(self, other: Point) -> bool:
        return self.x > other.x and self.y > other.y

    def __ge__(self, other: Point) -> bool:
        return self.x >= other.x and self.y >= other.y

    def __matmul__(self, other: Point) -> float:
        return self.x * other.x + self.y * other.y

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


Vec2 = Point
type Edge = tuple[Point, Point]


@dataclass(slots=True)
class Circle:
    base: Point
    radius: float


@dataclass(slots=True)
class AABB:
    start: Point
    end: Point

    def __repr__(self) -> str:
        return f"{self.start} -> {self.end}"

    def contains(self, point: Point) -> bool:
        return self.start.x <= point.x <= self.end.x and self.start.y <= point.y <= self.end.y


@dataclass(slots=True)
class OBB:
    center: Point
    half_extents: Vec2
    angle: float

    def __post_init__(self) -> None:
        if not (-np.pi <= self.angle <= np.pi):
            raise ValueError("`angle` must be in [-pi, pi].")
        if self.half_extents.x < 0 or self.half_extents.y < 0:
            raise ValueError("`half_extents` must be non-negative.")

    def contains(self, point: Point) -> bool:
        """Return whether `point` lies inside or on the OBB boundary."""
        offset = point - self.center
        cosine = math.cos(self.angle)
        sine = math.sin(self.angle)
        local_x = cosine * offset.x + sine * offset.y
        local_y = -sine * offset.x + cosine * offset.y
        return abs(local_x) <= self.half_extents.x and abs(local_y) <= self.half_extents.y

    def corners(self) -> list[Point]:
        """Return the four OBB corners in world coordinates."""
        axis = Point(math.cos(self.angle), math.sin(self.angle))
        normal = Point(-axis.y, axis.x)
        return [
            self.center - self.half_extents.x * axis - self.half_extents.y * normal,
            self.center + self.half_extents.x * axis - self.half_extents.y * normal,
            self.center + self.half_extents.x * axis + self.half_extents.y * normal,
            self.center - self.half_extents.x * axis + self.half_extents.y * normal,
        ]

    def enclosing_aabb(self) -> AABB:
        corners = self.corners()
        return AABB(
            Point(min(point.x for point in corners), min(point.y for point in corners)),
            Point(max(point.x for point in corners), max(point.y for point in corners)),
        )

    def __repr__(self) -> str:
        result = ""
        for i, corner in enumerate(self.corners()):
            result += repr(corner)
            if i != 0 or i != 3:
                result += " -> "
        return result


def make_aabb_from_points(points: list[Point]) -> AABB:
    start_x = min([p.x for p in points])
    start_y = min([p.y for p in points])
    end_x = max([p.x for p in points])
    end_y = max([p.y for p in points])
    return AABB(Point(start_x, start_y), Point(end_x, end_y))


def make_aabb_from_circle(circle: Circle) -> AABB:
    offset = Point(circle.radius, circle.radius)
    return AABB(circle.base - offset, circle.base + offset)


def make_obb_from_edge(edge: Edge, width: float) -> OBB:
    """Return the tight OBB around an edge with the given width."""
    if width < 0:
        raise ValueError("`width` must be non-negative.")
    delta: Vec2 = edge[1] - edge[0]
    angle = math.atan2(delta.y, delta.x)
    length = math.hypot(delta.x, delta.y)
    return OBB(
        (edge[0] + edge[1]) * 0.5,
        Point(length / 2, width / 2),
        angle,
    )


def dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def distance_squared(a: Point, b: Point) -> float:
    return dot(a - b, a - b)


def orientation(a: Point, b: Point, c: Point) -> float:
    """Positive = CCW (left), negative = CW (right), 0 = collinear."""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def point_to_edge_distance_squared(p: Point, e: Edge, epsilon: float = 1e-6) -> float:
    AB: Vec2 = e[1] - e[0]
    AP: Vec2 = p - e[0]
    if dot(AB, AB) < epsilon:  # Degenerate case
        return dot(AP, AP)
    t = max(0, min(1, dot(AP, AB) / dot(AB, AB)))
    closest = e[0] + t * AB
    return distance_squared(p, closest)


def ordered_vertices(edges: list[Edge], direction: str, directed: bool = False) -> list[Point]:
    if directed:
        return ordered_vertices_directed(edges, direction)
    if direction.lower() not in ("cw", "ccw"):
        raise ValueError("direction must be either 'cw' or 'ccw'.")
    if not edges:
        return []
    neighbors: dict[Point, list[Point]] = defaultdict(list)
    for p, q in edges:
        neighbors[p].append(q)
        neighbors[q].append(p)
    if any(len(ns) != 2 for ns in neighbors.values()):
        raise ValueError("Edges must form one simple closed cycle.")

    start = next(iter(neighbors))
    vertices = [start]
    previous: Point | None = None
    current = start
    while True:
        candidates = neighbors[current]
        next_vertex = candidates[0] if candidates[0] != previous else candidates[1]
        if next_vertex == start:
            break
        vertices.append(next_vertex)
        previous, current = current, next_vertex
        if len(vertices) > len(neighbors):
            raise ValueError("Edges do not form a simple cycle.")
    if len(vertices) != len(neighbors):
        raise ValueError("Edges contain disconnected components.")

    # Signed shoelace area
    area2 = sum(p.x * q.y - p.x * q.y for p, q in zip(vertices, vertices[1:] + vertices[:1]))
    if area2 == 0:
        raise ValueError("Vertices are collinear or form zero area.")

    is_ccw = area2 > 0
    if (direction.lower() == "ccw" and not is_ccw) or (direction.lower() == "cw" and is_ccw):
        vertices.reverse()
    return vertices


def ordered_vertices_directed(edges: list[Edge], direction: str) -> list[Point]:
    """WARNING: this algorithm does not work for collinear edges."""
    if direction.lower() not in ("cw", "ccw"):
        raise ValueError("direction must be either 'cw' or 'ccw'.")
    if not edges:
        return []
    successor: dict[Point, Point] = {}
    indegree: dict[Point, int] = {}

    for source, target in edges:
        if source in successor:
            raise ValueError(f"Vertex {source!r} has multiple outgoing edges.")
        successor[source] = target
        indegree[target] = indegree.get(target, 0) + 1
        indegree.setdefault(source, 0)
    if any(count != 1 for count in indegree.values()):
        raise ValueError("Each vertex must have exactly one incoming edge.")

    start = next(iter(successor))
    vertices: list[Point] = []
    current = start
    visited: set[Point] = set()
    while current not in visited:
        visited.add(current)
        vertices.append(current)
        if current not in successor:
            raise ValueError("Edges do not form a closed directed cycle.")
        current = successor[current]
    if current != start or len(vertices) != len(successor):
        raise ValueError("Edges must form one directed cycle containing all vertices.")

    # Signed shoelace area
    area2 = sum(p.x * q.y - p.y * q.x for p, q in zip(vertices, vertices[1:] + vertices[:1]))
    if area2 == 0:
        raise ValueError("Cycle has zero signed area.")

    is_ccw = area2 > 0
    if (direction.lower() == "ccw" and not is_ccw) or (direction.lower() == "cw" and is_ccw):
        vertices.reverse()
    return vertices


def aabb_intersects(first: AABB, second: AABB) -> bool:
    return not (
        first.end.x < second.start.x
        or first.start.x > second.end.x
        or first.end.y < second.start.y
        or first.start.y > second.end.y
    )


def aabb_intersects_obb_sat(aabb: AABB, obb: OBB) -> bool:
    """Return whether an AABB and an OBB intersect using SAT."""
    aabb_center_x = (aabb.start.x + aabb.end.x) / 2
    aabb_center_y = (aabb.start.y + aabb.end.y) / 2
    aabb_half_x = (aabb.end.x - aabb.start.x) / 2
    aabb_half_y = (aabb.end.y - aabb.start.y) / 2

    offset_x = obb.center.x - aabb_center_x
    offset_y = obb.center.y - aabb_center_y
    cosine = math.cos(obb.angle)
    sine = math.sin(obb.angle)
    axes = ((1.0, 0.0), (0.0, 1.0), (cosine, sine), (-sine, cosine))

    for axis_x, axis_y in axes:
        distance = abs(offset_x * axis_x + offset_y * axis_y)
        aabb_radius = aabb_half_x * abs(axis_x) + aabb_half_y * abs(axis_y)
        obb_radius = obb.half_extents.x * abs(cosine * axis_x + sine * axis_y)
        obb_radius += obb.half_extents.y * abs(-sine * axis_x + cosine * axis_y)
        if distance > aabb_radius + obb_radius:
            return False

    return True
