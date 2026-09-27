from ..geometry import Edge, Point, ordered_vertices, orientation


def slow_convex_hull(points: list[Point], direction: str = "ccw") -> list[Point]:
    """This runs in O(n^3)."""
    edges: list[Edge] = []
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            is_edge = True
            for k in range(len(points)):
                if k == i or k == j:
                    continue
                # NOTE: this should handle the degeneracy case (collinear)
                if orientation(points[i], points[j], points[k]) < 0:
                    is_edge = False
            if is_edge:
                edges.append((points[i], points[j]))
    return ordered_vertices(edges, direction, directed=True)


def convex_hull(points: list[Point]) -> list[Point]:
    """
    Compute the convex hull of a set of points with Graham's scan algorithm.
    This runs in O(n*logn), bound by the sorting step (O(n) otherwise).
    """
    _points = points.copy()
    # Lexicographic sort to handle the degeneracy case (same x-coordinate)
    _points.sort(key=lambda p: (p.x, p.y))
    upper_hull: list[Point] = [_points[0], _points[1]]
    for i in range(2, len(_points)):
        upper_hull.append(_points[i])
        while len(upper_hull) > 2 and orientation(*upper_hull[-3:]) >= 0:
            del upper_hull[-2]
    lower_hull: list[Point] = [_points[-1], _points[-2]]
    for i in range(len(points) - 2, -1, -1):
        lower_hull.append(_points[i])
        while len(lower_hull) > 2 and orientation(*lower_hull[-3:]) >= 0:
            del lower_hull[-2]
    lower_hull = lower_hull[1:-1]
    return upper_hull + lower_hull
