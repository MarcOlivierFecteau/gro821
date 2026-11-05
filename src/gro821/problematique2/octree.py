"""Octree acceleration structure for finite scene objects."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from gro821.problematique2.aabb import AABB, intersect_aabb, object_aabb
from gro821.problematique2.mesh import Mesh


@dataclass(slots=True)
class OctreeNode:
    bounds: AABB
    object_indices: list[int]
    children: list[OctreeNode]


@dataclass(slots=True)
class Octree:
    root: OctreeNode


def _child_bounds(bounds: AABB) -> list[AABB]:
    minimum, maximum = bounds
    midpoint = (minimum + maximum) / 2
    children: list[AABB] = []
    for x in (minimum[0], midpoint[0]):
        for y in (minimum[1], midpoint[1]):
            for z in (minimum[2], midpoint[2]):
                child_minimum = np.array([x, y, z], dtype=np.float32)
                child_maximum = np.array(
                    [
                        midpoint[0] if x == minimum[0] else maximum[0],
                        midpoint[1] if y == minimum[1] else maximum[1],
                        midpoint[2] if z == minimum[2] else maximum[2],
                    ],
                    dtype=np.float32,
                )
                children.append((child_minimum, child_maximum))
    return children


def _build_node(
    bounds: AABB,
    object_indices: list[int],
    object_bounds: dict[int, AABB],
    max_objects: int,
    max_depth: int,
    depth: int,
) -> OctreeNode:
    node = OctreeNode(bounds, [], [])
    if depth >= max_depth or len(object_indices) <= max_objects:
        node.object_indices = object_indices
        return node

    child_bounds = _child_bounds(bounds)
    child_indices: list[list[int]] = [[] for _ in child_bounds]
    for index in object_indices:
        object_minimum, object_maximum = object_bounds[index]
        containing_child = next(
            (
                child_index
                for child_index, (child_minimum, child_maximum) in enumerate(child_bounds)
                if np.all(object_minimum >= child_minimum)
                and np.all(object_maximum <= child_maximum)
            ),
            None,
        )
        if containing_child is None:
            node.object_indices.append(index)
        else:
            child_indices[containing_child].append(index)

    if not any(child_indices):
        node.object_indices = object_indices
        return node

    node.children = [
        _build_node(
            child_bounds[index],
            indices,
            object_bounds,
            max_objects,
            max_depth,
            depth + 1,
        )
        for index, indices in enumerate(child_indices)
        if indices
    ]
    return node


def build_octree_from_bounds(
    root_bounds: AABB,
    bounds_list: list[AABB],
    max_objects: int = 8,
    max_depth: int = 8,
) -> Octree:
    """Build an octree over a list of AABBs (one per primitive) inside `root_bounds`."""
    if max_objects < 1:
        raise ValueError("max_objects must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    object_bounds = dict(enumerate(bounds_list))
    return Octree(
        _build_node(root_bounds, list(object_bounds), object_bounds, max_objects, max_depth, 0)
    )


def build_octree(
    objects: list[dict[str, Any]],
    max_objects: int = 8,
    max_depth: int = 8,
) -> Octree | None:
    """
    Build an octree containing every finite object in ``objects``.

    `Mesh` Objects contribute one entry per face; leaf entries are primitive
    indices in global traversal order (mesh faces first).
    """
    if max_objects < 1:
        raise ValueError("max_objects must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    bounds_list: list[AABB] = []
    for obj in objects:
        match obj["type"]:
            case "mesh":
                mesh: Mesh = obj["mesh"]
                bounds_list.extend(zip(mesh.bounds_min, mesh.bounds_max))
            case "plane":
                continue
            case _:
                b = object_aabb(obj)
                if b is not None:
                    bounds_list.append(b)
    if not bounds_list:
        return None

    # Root region: the union box of all primitives (octree partitions this
    # box, not an arbitrary world volume).
    root_box = (
        np.min(np.stack([b[0] for b in bounds_list]), axis=0),
        np.max(np.stack([b[1] for b in bounds_list]), axis=0),
    )
    return build_octree_from_bounds(root_box, bounds_list, max_objects, max_depth)


def query_octree(
    tree: Octree | None,
    origin: np.ndarray,
    direction: np.ndarray,
    t_max: float = np.inf,
) -> list[int]:
    """Return finite object indices whose octree nodes intersect a ray."""
    if tree is None:
        return []
    if intersect_aabb(origin, direction, tree.root.bounds) is None:
        return []

    object_indices: list[int] = []
    nodes = [tree.root]
    while nodes:
        node = nodes.pop()
        # == Distance pruning
        interval = intersect_aabb(origin, direction, node.bounds)
        if interval is None:
            continue
        t_near, _ = interval
        if t_near > t_max:
            continue
        object_indices.extend(node.object_indices)
        nodes.extend(node.children)
    return object_indices
