"""Bounding volume hierarchy for finite scene objects."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from gro821.problematique2.aabb import AABB, intersect_aabb, object_aabb


@dataclass(slots=True)
class BVHNode:
    bounds: AABB
    object_indices: list[int]
    children: tuple[BVHNode, BVHNode] | None = None


@dataclass(slots=True)
class BVH:
    root: BVHNode


def _merge_bounds(bounds: list[AABB]) -> AABB:
    return (
        np.min(np.stack([minimum for minimum, _ in bounds]), axis=0),
        np.max(np.stack([maximum for _, maximum in bounds]), axis=0),
    )


def _build_node(
    object_indices: list[int],
    object_bounds: dict[int, AABB],
    max_objects: int,
    max_depth: int,
    depth: int,
) -> BVHNode:
    bounds = _merge_bounds([object_bounds[index] for index in object_indices])
    if len(object_indices) <= max_objects or depth >= max_depth:
        return BVHNode(bounds, object_indices)

    centroids = np.stack(
        [
            (object_bounds[index][0] + object_bounds[index][1]) / 2
            for index in object_indices
        ]
    )
    split_axis = int(np.argmax(np.ptp(centroids, axis=0)))
    ordering = np.argsort(centroids[:, split_axis], kind="stable")
    midpoint = len(object_indices) // 2
    left_indices = [object_indices[index] for index in ordering[:midpoint]]
    right_indices = [object_indices[index] for index in ordering[midpoint:]]
    children = (
        _build_node(left_indices, object_bounds, max_objects, max_depth, depth + 1),
        _build_node(right_indices, object_bounds, max_objects, max_depth, depth + 1),
    )
    return BVHNode(bounds, [], children)


def build_bvh(
    objects: list[dict[str, Any]],
    max_objects: int = 4,
    max_depth: int = 32,
) -> BVH | None:
    """Build a median-split BVH containing every finite object."""
    if max_objects < 1:
        raise ValueError("max_objects must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    object_bounds = {
        index: bounds
        for index, obj in enumerate(objects)
        if (bounds := object_aabb(obj)) is not None
    }
    if not object_bounds:
        return None
    return BVH(
        _build_node(
            list(object_bounds),
            object_bounds,
            max_objects,
            max_depth,
            0,
        )
    )


def query_bvh(
    tree: BVH | None,
    origin: np.ndarray,
    direction: np.ndarray,
    t_max: float = np.inf,
) -> list[int]:
    """Return finite object indices whose BVH bounds intersect a ray."""
    if tree is None:
        return []
    if intersect_aabb(origin, direction, tree.root.bounds) is None:
        return []

    object_indices: list[int] = []
    nodes = [tree.root]
    while nodes:
        node = nodes.pop()
        # == Distance pruning (compute)
        # Prune entire subtrees if they are beyond the distance we care about.
        interval = intersect_aabb(origin, direction, node.bounds)
        if interval is None:
            continue
        t_near, _ = interval
        if t_near > t_max: # Subtree is beyond: prune it
            continue
        if node.children is None:
            object_indices.extend(node.object_indices)
        else:
            nodes.extend(node.children)
    return object_indices
