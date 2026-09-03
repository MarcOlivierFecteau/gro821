from collections import defaultdict
from dataclasses import dataclass
from typing import Final

import numpy as np

EPSILON: Final = 1e-9


@dataclass
class Point:
    x: float
    y: float

    def to_np(self):
        return np.array([self.x, self.y], dtype=np.float64)

    def __abs__(self):
        return Point(abs(self.x), abs(self.y))

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __add__(self, other: Point):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Point):
        return Point(self.x * other.x, self.y * other.y)

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

    def __matmul__(self, other: Point):
        return self.x * other.x + self.y * other.y


type Edge = tuple[Point, Point]


@dataclass
class Circle:
    base: Point
    radius: float


def dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def distance_squared(a: Point, b: Point) -> float:
    return dot(a - b, a - b)


def orientation(a: Point, b: Point, c: Point) -> float:
    """Positive = CCW (left), negative = CW (right), 0 = collinear."""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


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
