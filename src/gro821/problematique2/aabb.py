"""Axis-aligned bounding boxes used to reject rays before primitive tests."""

from typing import Any

import numpy as np

type AABB = tuple[np.ndarray, np.ndarray]


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
        case _:
            raise ValueError(f"Unsupported object type: {obj['type']}.")


def root_aabb(objects: list[dict[str, Any]]) -> AABB | None:
    """Return the AABB enclosing all finite objects in a scene."""
    bounds = [bounds for obj in objects if (bounds := object_aabb(obj)) is not None]
    if not bounds:
        return None
    return (
        np.min(np.stack([minimum for minimum, _ in bounds]), axis=0),
        np.max(np.stack([maximum for _, maximum in bounds]), axis=0),
    )


def intersects_aabb(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds: AABB,
) -> bool:
    """Return whether a forward ray intersects an AABB."""
    minimum, maximum = bounds
    parallel = direction == 0
    parallel_outside = parallel & ((origin < minimum) | (origin > maximum))
    if np.any(parallel_outside):
        return False

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
    return bool(np.max(near) <= np.min(far) and np.min(far) >= 0.0)
