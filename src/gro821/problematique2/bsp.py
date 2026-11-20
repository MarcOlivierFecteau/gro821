"""Bounding volume hierarchy for finite scene objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from gro821.problematique2.aabb import AABB, intersect_aabb, object_aabb

type Axis = Literal[0, 1, 3]

@dataclass(slots=True)
class BSPNode:
    bounds: AABB
    object_indices: list[int]
    children: tuple[BSPNode, BSPNode] | None = None


@dataclass(slots=True)
class BSP:
    root: BSPNode


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
) -> BSPNode:
    bounds = _merge_bounds([object_bounds[index] for index in object_indices])

    if len(object_indices) <= max_objects or depth >= max_depth:
        return BSPNode(bounds, object_indices)

    centroids = np.stack(
        [(object_bounds[index][0] + object_bounds[index][1]) / 2 for index in object_indices]
    )

    #On prend l'axe le plus long, et on trouve la médiane
    split_axis = int(np.argmax(np.ptp(centroids, axis=0)))
    median = float(np.median(centroids[:, split_axis]))

    frt_indices, bck_indices = [], []

    for idx in object_indices:
        mini, maxi = object_bounds[idx]
        #si l'objet est plus grande que la mediane, alors elle va en avant
        if mini[split_axis] >= median:
            frt_indices.append(idx)
        #si l'objet est plus petit que la médiane, alors elle va en arrière
        elif maxi[split_axis] <= median:
            bck_indices.append(idx)
        else:
            frt_indices.append(idx)
            bck_indices.append(idx)

    children = (
        _build_node(frt_indices, object_bounds, max_objects, max_depth, depth + 1),
        _build_node(bck_indices, object_bounds, max_objects, max_depth, depth + 1),
    )



    return BSPNode(bounds, [], children)


def build_bsp_from_bounds(
    bounds_list: list[AABB],
    max_objects: int = 4,
    max_depth: int = 32,
) -> BSP:
    """Build a median-split BSP over a list of AABBs (one per primitive)."""
    object_bounds = dict(enumerate(bounds_list))
    return BSP(_build_node(list(object_bounds), object_bounds, max_objects, max_depth, 0))


def build_bsp(
    objects: list[dict[str, Any]],
    max_objects: int = 4,
    max_depth: int = 32,
) -> BSP | None:
    """
    Build a median-split BSP containing every finite object.

    `Mesh` objects contribute one leaf entry per face.
    """
    if max_objects < 1:
        raise ValueError("max_objects must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    bounds_list: list[AABB] = []
    for obj in objects:
        match obj["type"]:
            case "mesh":
                mesh = obj["mesh"]
                bounds_list.extend(zip(mesh.bounds_min, mesh.bounds_max))
            case "plane":
                continue
            case _:
                b = object_aabb(obj)
                if b is not None:
                    bounds_list.append(b)
    if not bounds_list:
        return None
    return build_bsp_from_bounds(bounds_list, max_objects, max_depth)


def query_bsp(
    tree: BSP | None,
    origin: np.ndarray,
    direction: np.ndarray,
    t_max: float = np.inf,
) -> list[int]:
    """Return finite object indices whose BSP bounds intersect a ray."""
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
        if t_near > t_max:  # Subtree is beyond: prune it
            continue
        if node.children is None:
            object_indices.extend(node.object_indices)
        else:
            nodes.extend(node.children)
    return object_indices
