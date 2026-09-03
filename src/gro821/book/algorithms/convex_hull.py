from ..geometry import Edge, Point, ordered_vertices, orientation


def slow_convex_hull(points: list[Point], direction: str = "ccw") -> list[Point]:
    edges: list[Edge] = []
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            valid = True
            for k in range(len(points)):
                if k == i or k == j:
                    continue
                if orientation(points[i], points[j], points[k]) > 0:
                    valid = False
            if valid:
                edges.append((points[i], points[j]))
    return ordered_vertices(edges, direction, directed=True)
