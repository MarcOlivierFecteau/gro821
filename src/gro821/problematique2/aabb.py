"""Axis-aligned bounding boxes used to reject rays before primitive tests."""

from typing import Any

import numpy as np

from gro821.problematique2 import aabb_intersection_test_count
from gro821.problematique2.mesh import Mesh
from gro821.sandbox import fmt
from gro821.sandbox.fmt import ANSI

type AABB = tuple[np.ndarray, np.ndarray]
type MeshFace = tuple[int, int]


def transform_aabb(bounds: AABB, transform: np.ndarray) -> AABB:
    minimum, maximum = bounds
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float32,
    )
    transformed = (transform @ corners.T).T[:, :3]
    return transformed.min(axis=0), transformed.max(axis=0)


def object_aabb(obj: dict[str, Any]) -> AABB | None:
    """Return an object's finite AABB, or ``None`` for unbounded objects."""
    match obj["type"]:
        case "sphere":
            center = np.asarray(obj["position"], dtype=np.float32)
            radius = float(obj["radius"])
            extent = np.full(3, radius, dtype=np.float32)
            return center - extent, center + extent
        case "triangle":
            p1 = np.asarray(obj["p1"], dtype=np.float32)
            points = np.stack((p1, p1 + obj["e1"], p1 + obj["e2"]))
            return np.min(points, axis=0), np.max(points, axis=0)
        case "plane":
            return None
        case "mesh":
            return transform_aabb(mesh_aabb(obj["mesh"]), obj["transform"])
        case _:
            raise ValueError(f"Unsupported object type: {obj['type']}.")


def mesh_aabb(mesh: Mesh) -> AABB:
    """AABB enclosing an entire mesh (union of per-face boxes)."""
    return mesh.bounds_min.min(axis=0), mesh.bounds_max.max(axis=0)


def mesh_object_aabbs(obj: dict[str, Any], mesh_index: int) -> dict[MeshFace, AABB]:
    """Return world-space AABBs indexed by mesh instance and face."""
    mesh: Mesh = obj["mesh"]
    transform = obj["transform"]
    return {
        (mesh_index, face_index): transform_aabb(
            (mesh.bounds_min[face_index], mesh.bounds_max[face_index]), transform
        )
        for face_index in range(len(mesh.faces))
    }


def root_aabb(objects: list[dict[str, Any]]) -> AABB | None:
    """Return the AABB enclosing all finite objects in a scene."""
    bounds = [bounds for obj in objects if (bounds := object_aabb(obj)) is not None]
    if not bounds:
        return None
    return (
        np.min(np.stack([minimum for minimum, _ in bounds]), axis=0),
        np.max(np.stack([maximum for _, maximum in bounds]), axis=0),
    )


def intersect_aabb(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds: AABB,
) -> tuple[float, float] | None:
    """Return whether a forward ray intersects an AABB."""
    global aabb_intersection_test_count
    aabb_intersection_test_count += 1

    minimum, maximum = bounds
    parallel = direction == 0
    parallel_outside = parallel & ((origin < minimum) | (origin > maximum))
    if np.any(parallel_outside):
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_direction = np.divide(
            1.0,
            direction,
            out=np.zeros_like(direction, dtype=np.float64),
            where=~parallel,
        )
        distances_a = (minimum - origin) * inverse_direction
        distances_b = (maximum - origin) * inverse_direction

    near = np.where(parallel, -np.inf, np.minimum(distances_a, distances_b))
    far = np.where(parallel, np.inf, np.maximum(distances_a, distances_b))
    # == Distance pruning
    # Promoting returned value to hit interval instead of boolean.
    # Note that casting to float to compute `t_near > t_far` on scalars once
    # instead of via NumPy reductions on combined boolean: (marginally) cheaper.
    t_near = float(np.max(near))
    t_far = float(np.min(far))
    if t_near > t_far or t_far < 0.0:
        return None
    return t_near, t_far


def pprint_aabb_tests(width, height):
    global aabb_intersection_test_count

    def tests_per_pixel(count: int, width, height, out: type = str) -> float | str:
        result = count / (width * height)
        if out == str:
            return f"{result:.2f} / px"
        return out(result)

    print(
        f"\tAABB: {aabb_intersection_test_count} "
        f"({fmt.color(ANSI.DIM, tests_per_pixel(aabb_intersection_test_count, width, height))})"
    )
    return aabb_intersection_test_count
