"""
Re-implementation of `raytracing.py` from the provided codebase.
"""

from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Final

import numpy as np
from PIL import Image

import gro821.problematique2.aabb as aabb_module
from gro821.problematique2 import *
from gro821.problematique2 import bunny
from gro821.problematique2.aabb import (
    AABB,
    MeshFace,
    intersect_aabb,
    mesh_object_aabbs,
    pprint_aabb_tests,
    root_aabb,
)
from gro821.problematique2.bvh import BVH, BVHNode, build_bvh
from gro821.problematique2.mesh import Mesh, build_mesh
from gro821.problematique2.octree import Octree, OctreeNode, build_octree, query_octree
from gro821.problematique2.transforms import flip_yz, rotation_y, rotation_z, translation
from gro821.sandbox import fmt
from gro821.sandbox.fmt import ANSI

type Vector = np.ndarray[tuple[int], np.dtype[np.float32]]
Point = Vertex = Vector

type Vec3 = tuple[float, float, float]
Point3 = Vec3
type RGB = tuple[float, float, float]

type Intersection = tuple[int, tuple[float, float, float]]
type Hit = tuple[str, Any]  # ("mesh", (mesh_index, t, u, v, face_index)) | ...


class ObjectType(StrEnum):
    PLANE = "plane"
    SPHERE = "sphere"
    TRIANGLE = "triangle"
    MESH = "mesh"


type Object = dict[str, Any]

# Global parameters
plane_sdf_half_size: Final = 0.5
plane_sdf_band_width: Final = 0.5
plane_orange_palette: Final = (
    # np.array([255 / 255, 127 / 255, 0 / 255], dtype=np.float32),
    # np.array([127 / 255, 70 / 255, 0 / 255], dtype=np.float32),
    np.array([1.0, 1.0, 1.0], dtype=np.float32),
    np.array([0.5, 0.5, 0.5], dtype=np.float32),
)
light_pos: Final = np.array([5, 5, 10], dtype=np.float32)
light_color = np.array([1, 1, 1], dtype=np.float32)
PINK: Final = (230 / 255, 125 / 255, 246 / 255)

glob_ambient: Final = 0.05
glob_diffuse_coeff: Final = 1.0
glob_specular_coeff: Final = 1.0
glob_specular_k: Final = 50

max_depth: Final = 2  # Upper limit on number of reflections

ASPECT_RATIO: Final = 16 / 10
HEIGHT: Final = 1600
WIDTH: Final = int(HEIGHT * ASPECT_RATIO)

SHADING = "gouraud"  # "flat" | "gouraud"

# Ray rejection method
ACCELERATION = "bvh"  # "root_aabb" | "octree" | "bvh"

CULLING: Final = True
EARLY_EXIT: Final = True
VECTORIZED: Final = True
TILE_SIZE: Final = 32
MULTIPROC: Final = True

type RenderState = tuple[
    list[Object],
    Point,
    AABB | None,
    Octree | None,
    BVH | None,
    Point,
    np.ndarray,
]

_worker_render_state: RenderState | None = None


def normalize(x: Vector):
    x /= np.linalg.norm(x)
    return x


def transform_point(transform: np.ndarray, point: Point) -> Point:
    homogeneous = np.append(point, 1.0)
    return (transform @ homogeneous)[:3]


def transform_vector(transform: np.ndarray, vector: Vector) -> Vector:
    return transform[:3, :3] @ vector


def intersect_plane(origin: Point, direction: Vector, plane: Point, normal: Vector) -> float:
    global plane_intersection_test_count
    plane_intersection_test_count += 1

    den = np.dot(direction, normal)
    if np.abs(den) <= 1e-6:
        return np.inf
    distance = np.dot(plane - origin, normal) / den
    return float(distance) if distance >= 0 else np.inf


def intersect_sphere(origin: Point, direction: Vector, sphere: Point, radius: float) -> float:
    global sphere_intersection_test_count
    sphere_intersection_test_count += 1

    a = np.dot(direction, direction)
    dist_to_sphere = origin - sphere
    b: np.float64 = 2 * np.dot(direction, dist_to_sphere)
    c: np.float64 = np.dot(dist_to_sphere, dist_to_sphere) - radius * radius
    det = b * b - 4 * a * c
    if det <= 0:
        return np.inf
    dist_sqrt = np.sqrt(det)
    q = (-b + (dist_sqrt if b >= 0 else -dist_sqrt)) / 2
    t0 = q / a
    t1 = c / q
    t0, t1 = min(t0, t1), max(t0, t1)
    if t1 >= 0:
        return t1 if t0 < 0 else t0
    return np.inf


def intersect_triangle(
    origin: Point,
    direction: Vector,
    triangle: Object,
) -> tuple[float, float, float]:
    """
    Möller-Trumbore intersection algorithm. Also returns `u` and `v` weights
    for barycentric coordinates.
    """
    global triangle_intersection_test_count
    triangle_intersection_test_count += 1

    EPS: Final = 1e-6
    # == Möller-Trumbore without redundant work (compute)
    # Optimized version of Möller-Trumbore.
    e1, e2 = triangle["e1"], triangle["e2"]
    p1 = triangle["p1"]

    # Use the standard Möller-Trumbore edge order
    pv = np.cross(direction, e2)
    det = np.dot(e1, pv)
    # == Back-face culling (compute)
    # With this edge order and the mesh winding, front-facing triangles have
    # a positive determinant. Without culling, only reject degenerate triangles.
    if CULLING:
        if det <= EPS:
            return np.inf, 0.0, 0.0
    elif abs(det) < EPS:
        return np.inf, 0.0, 0.0

    idet = 1.0 / det
    tv = origin - p1

    u = np.dot(tv, pv) * idet
    if u < 0 or u > 1:
        return np.inf, 0.0, 0.0

    qv = np.cross(tv, e1)
    v = np.dot(direction, qv) * idet
    if v < 0 or u + v > 1:
        return np.inf, 0.0, 0.0

    return np.dot(e2, qv) * idet, u, v


def intersect_triangles_batch(
    mesh: Mesh,
    origin: Point,
    direction: Vector,
    face_indices: list[int] | np.ndarray,
    t_max: float = np.inf,
    exclude_face: int | None = None,
) -> tuple[float, float, float, int] | None:
    """Vectorized Möller-Trumbore over a batch of faces.

    Returns (t, u, v, face_idx) for the closest valid hit within `t_max`,
    or None. Applies back-face culling via the determinant sign.
    """
    EPS: Final = 1e-6
    fi = np.asarray(face_indices, dtype=np.int64)
    if exclude_face is not None:
        fi = fi[fi != exclude_face]
    if fi.size == 0:
        return None

    e1 = mesh.edge1[fi]  # (K, 3) -- One fancy-index per field
    e2 = mesh.edge2[fi]
    p1 = mesh.p1[fi]

    pv = np.cross(direction, e2)  # (K, 3) -- Broadcast over rows
    det = np.einsum("ij,ij->i", e1, pv)  # (K,) -- row-wise dot
    tv = origin - p1  # (K, 3) -- Broadcast
    u = np.einsum("ij,ij->i", tv, pv) / det
    qv = np.cross(tv, e1)
    v = np.einsum("j,ij->i", direction, qv) / det
    t = np.einsum("ij,ij->i", e2, qv) / det

    # Match the scalar intersection path's determinant handling.
    not_culled = det > EPS if CULLING else np.abs(det) >= EPS
    valid = (
        not_culled  # non-degenerate; scalar mode applies the same culling policy
        & (u >= 0.0)
        & (v >= 0.0)
        & (u + v <= 1.0)
        & (t >= 0.0)
        & (t < t_max)
    )

    if not np.any(valid):
        return None
    t_masked = np.where(valid, t, np.inf)
    best = int(np.argmin(t_masked))
    return t_masked[best], u[best], v[best], fi[best]


def any_triangle_hit_batch(
    mesh: Mesh,
    origin: Point,
    direction: Vector,
    face_indices: list[int] | np.ndarray,
    t_max: float = np.inf,
    exclude_face: int | None = None,
) -> bool:
    """Return whether any face in a batch intersects before ``t_max``.

    This is specialized for shadow rays: unlike ``intersect_triangles_batch``,
    it does not compute barycentric coordinates or select the closest hit for
    faces that have already failed an earlier intersection test.
    """
    EPS: Final = 1e-6
    fi = np.asarray(face_indices, dtype=np.int64)
    if exclude_face is not None:
        fi = fi[fi != exclude_face]
    if fi.size == 0:
        return False

    e1 = mesh.edge1[fi]
    e2 = mesh.edge2[fi]
    p1 = mesh.p1[fi]

    pv = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    valid = det > EPS if CULLING else np.abs(det) >= EPS
    if not np.any(valid):
        return False

    e1 = e1[valid]
    e2 = e2[valid]
    p1 = p1[valid]
    pv = pv[valid]
    det = det[valid]

    tv = origin - p1
    u = np.einsum("ij,ij->i", tv, pv) / det
    valid = (u >= 0.0) & (u <= 1.0)
    if not np.any(valid):
        return False

    e1 = e1[valid]
    e2 = e2[valid]
    tv = tv[valid]
    u = u[valid]
    pv = pv[valid]
    det = det[valid]

    qv = np.cross(tv, e1)
    v = np.einsum("j,ij->i", direction, qv) / det
    valid = (v >= 0.0) & (u + v <= 1.0)
    if not np.any(valid):
        return False

    e2 = e2[valid]
    qv = qv[valid]
    det = det[valid]
    t = np.einsum("ij,ij->i", e2, qv) / det
    return bool(np.any((t >= 0.0) & (t < t_max)))


def intersect_triangle_mesh(
    mesh: Mesh,
    origin: Point,
    direction: Vector,
    face_idx: int,
    t_max: float = np.inf,
) -> tuple[float, float, float] | None:
    """
    Scalar Möller-Trumbore for a single mesh face.

    Returns (t, u, v) within t_max, or None. Same fields and same algorithm as
    `intersect_triangles_batch`, but one face at a time (like before VECTORIZED).
    """
    global triangle_intersection_test_count
    triangle_intersection_test_count += 1

    EPS: Final = 1e-6

    e1 = mesh.edge1[face_idx]
    e2 = mesh.edge2[face_idx]
    p1 = mesh.p1[face_idx]

    pv = np.cross(direction, e2)
    det = np.dot(e1, pv)
    # == Back-face culling
    if CULLING:
        if det <= EPS:
            return None
    elif abs(det) < EPS:
        return None

    idet = 1.0 / det
    tv = origin - p1
    u = np.dot(tv, pv) * idet
    if u < 0.0 or u > 1.0:
        return None

    qv = np.cross(tv, e1)
    v = np.dot(direction, qv) * idet
    if v < 0.0 or u + v > 1.0:
        return None

    t = np.dot(e2, qv) * idet
    if t < 0.0 or t >= t_max:
        return None
    return t, u, v


def all_mesh_faces(scene: list[Object]) -> list[MeshFace]:
    return [
        (mesh_index, face_index)
        for mesh_index, obj in enumerate(scene)
        if obj["type"] == ObjectType.MESH
        for face_index in range(len(obj["mesh"].faces))
    ]


def intersect_mesh_candidates(
    scene: list[Object],
    candidates: list[MeshFace],
    origin: Point,
    direction: Vector,
    t_max: float = np.inf,
    exclude_face: MeshFace | None = None,
) -> tuple[int, float, float, float, int] | None:
    candidates_by_mesh: dict[int, list[int]] = {}
    for mesh_index, face_index in candidates:
        if (mesh_index, face_index) == exclude_face:
            continue
        candidates_by_mesh.setdefault(mesh_index, []).append(face_index)

    closest: tuple[int, float, float, float, int] | None = None
    for mesh_index, face_indices in candidates_by_mesh.items():
        mesh_obj = scene[mesh_index]
        mesh = mesh_obj["mesh"]
        inverse_transform = mesh_obj["inverse_transform"]
        local_origin = transform_point(inverse_transform, origin)
        local_direction = transform_vector(inverse_transform, direction)
        result = intersect_triangles_batch(
            mesh,
            local_origin,
            local_direction,
            face_indices,
            t_max,
            exclude_face=None,
        )
        if result is None:
            continue
        t, u, v, face_index = result
        if closest is None or t < closest[1]:
            closest = (mesh_index, t, u, v, face_index)
    return closest


def intersect(origin: Point, direction: Vector, obj: Object):
    global intersection_test_count
    intersection_test_count += 1

    match obj["type"]:
        case ObjectType.PLANE:
            return intersect_plane(origin, direction, obj["position"], obj["normal"]), 0.0, 0.0
        case ObjectType.SPHERE:
            return intersect_sphere(origin, direction, obj["position"], obj["radius"]), 0.0, 0.0
        case ObjectType.TRIANGLE:
            return intersect_triangle(origin, direction, obj)
        case _:
            raise ValueError(f"Unsupported object type: {obj['type']}.")


def intersect_bvh_closest(
    scene: list[Object],
    bvh: BVH | None,
    origin: Point,
    direction: Vector,
    t_max: float,
    exclude_face: MeshFace | None,
) -> tuple[int, float, float, float, int] | None:
    """Find the closest mesh hit with near-first stack-based BVH traversal."""
    if bvh is None:
        return None

    root_interval = intersect_aabb(origin, direction, bvh.root.bounds)
    if root_interval is None or root_interval[0] > t_max:
        return None

    best: tuple[int, float, float, float, int] | None = None
    best_t = t_max
    stack: list[tuple[float, BVHNode]] = [(root_interval[0], bvh.root)]

    while stack:
        node_near, node = stack.pop()
        if node_near >= best_t:
            continue

        if node.children is None:
            candidates_by_mesh: dict[int, list[int]] = {}
            for mesh_index, face_index in node.object_indices:
                if (mesh_index, face_index) == exclude_face:
                    continue
                candidates_by_mesh.setdefault(mesh_index, []).append(face_index)

            for mesh_index, face_indices in candidates_by_mesh.items():
                mesh_obj = scene[mesh_index]
                local_origin = transform_point(mesh_obj["inverse_transform"], origin)
                local_direction = transform_vector(mesh_obj["inverse_transform"], direction)
                if VECTORIZED:
                    global triangle_intersection_test_count
                    triangle_intersection_test_count += len(face_indices)
                    result = intersect_triangles_batch(
                        mesh_obj["mesh"],
                        local_origin,
                        local_direction,
                        face_indices,
                        best_t,
                    )
                    if result is None:
                        continue
                    t, u, v, face_index = result
                    if t < best_t:
                        best_t = t
                        best = (mesh_index, t, u, v, int(face_index))
                else:
                    for face_index in face_indices:
                        result = intersect_triangle_mesh(
                            mesh_obj["mesh"],
                            local_origin,
                            local_direction,
                            face_index,
                            best_t,
                        )
                        if result is None:
                            continue
                        t, u, v = result
                        if t < best_t:
                            best_t = t
                            best = (mesh_index, t, u, v, face_index)
            continue

        children: list[tuple[float, BVHNode]] = []
        for child in node.children:
            interval = intersect_aabb(origin, direction, child.bounds)
            if interval is not None and interval[0] < best_t:
                children.append((interval[0], child))
        children.sort(key=lambda item: item[0], reverse=True)
        stack.extend(children)

    return best


def intersect_scene(
    scene: list[Object],
    root_bounds: AABB | None,
    origin: Point,
    direction: Vector,
    octree: Octree | None = None,
    bvh: BVH | None = None,
    t_max: float = np.inf,
    exclude_face: MeshFace | None = None,
) -> list[Hit] | list[Intersection]:
    # == Selectable spatial acceleration (test)
    # Keep the root AABB baseline available for comparison while allowing the
    # other methods to reject objects using smaller node bounds.
    best: tuple[int, float, float, float, int] | None = None
    candidate_indices: list[MeshFace] | None = None
    if ACCELERATION == "bvh":
        best = intersect_bvh_closest(scene, bvh, origin, direction, t_max, exclude_face)
    elif ACCELERATION == "octree":
        candidate_indices = query_octree(octree, origin, direction, t_max)
    elif ACCELERATION == "root_aabb":
        visible = root_bounds is None or intersect_aabb(origin, direction, root_bounds) is not None
        candidate_indices = all_mesh_faces(scene) if visible else []
    else:
        raise ValueError(f"Unsupported acceleration mode: {ACCELERATION}.")

    hits: list[Hit] = []

    if best is not None:
        hits.append(("mesh", best))
    elif candidate_indices is not None and VECTORIZED:
        # One vectorized pass over all mesh candidates
        global triangle_intersection_test_count
        triangle_intersection_test_count += len(candidate_indices)
        best = intersect_mesh_candidates(
            scene, candidate_indices, origin, direction, t_max, exclude_face
        )
        if best is not None:
            hits.append(("mesh", best))
    elif candidate_indices is not None:
        # Scalar path: same tests, one face at a time.
        for mesh_index, face_index in candidate_indices:
            if (mesh_index, face_index) == exclude_face:
                continue
            mesh_obj = scene[mesh_index]
            local_origin = transform_point(mesh_obj["inverse_transform"], origin)
            local_direction = transform_vector(mesh_obj["inverse_transform"], direction)
            result = intersect_triangle_mesh(
                mesh_obj["mesh"], local_origin, local_direction, face_index, t_max
            )
            if result:
                hits.append(("mesh", (mesh_index, *result, face_index)))

    # Non-mesh objects, individually (plane, sphere)
    for obj in scene:
        match obj["type"]:
            case ObjectType.PLANE:
                t_obj = intersect_plane(origin, direction, obj["position"], obj["normal"])
                if t_obj < t_max:
                    hits.append(("plane", (t_obj, obj)))
            case ObjectType.SPHERE:
                t_obj = intersect_sphere(origin, direction, obj["position"], obj["radius"])
                if t_obj < t_max:
                    hits.append(("sphere", (t_obj, obj)))

    return hits


def mesh_normal(mesh: Mesh, face_idx: int, u: float, v: float) -> Vector:
    if SHADING == "flat":
        return mesh.normals[face_idx]
    # Gouraud: blend the three vertex normals with barycentric weights
    f = mesh.faces[face_idx]
    normal = (
        (1 - u - v) * mesh.vertex_normals[f[0]]
        + u * mesh.vertex_normals[f[1]]
        + v * mesh.vertex_normals[f[2]]
    )
    return normal / np.linalg.norm(normal)  # pyright: ignore[reportReturnType]


def get_normal(obj: Object, intersect_point: Point, u: float = 0.0, v: float = 0.0) -> Vector:
    match obj["type"]:
        case ObjectType.PLANE:
            # == Use the plane's geometric normal for shading
            plane_normal = np.asarray(obj["normal"], dtype=np.float32)
            return plane_normal / np.linalg.norm(plane_normal)
        case ObjectType.SPHERE:
            return normalize(intersect_point - obj["position"])
        case _:
            raise ValueError(f"Unsupported object type: {obj['type']} in `get_normal`.")
    raise RuntimeError()


def get_color(obj: Object, intersect_point: Point) -> Vector:
    color = obj["color"]
    if isinstance(color, np.ndarray):
        return color
    assert callable(color)
    return color(intersect_point)  # pyright: ignore[reportReturnType]


def trace_ray(
    scene: list[Object],
    light: Point,
    origin: Point,
    direction: Vector,
    root_bounds: AABB | None = None,
    octree: Octree | None = None,
    bvh: BVH | None = None,
):
    t = np.inf
    u = v = 0.0
    hit_mesh_index: int | None = None
    hit_face: int | None = None
    hit_obj: Object | None = None

    for kind, payload in intersect_scene(scene, root_bounds, origin, direction, octree, bvh):
        if kind == "mesh":
            mesh_index_hit, t_hit, u_hit, v_hit, face_hit = payload
        else:
            t_hit = payload[0]
        if t_hit < t:
            t = t_hit
            if kind == "mesh":
                u, v = u_hit, v_hit
                hit_mesh_index, hit_face = mesh_index_hit, face_hit
                hit_obj = None
            else:
                hit_obj = payload[1]  # pyright: ignore[reportAssignmentType]
                hit_mesh_index = None
                hit_face = None

    if t == np.inf:
        return None

    # Find point of insersect on object
    intersect_point = origin + direction * t
    if hit_mesh_index is not None and hit_face is not None:
        mesh_obj = scene[hit_mesh_index]
        local_normal = mesh_normal(mesh_obj["mesh"], hit_face, u, v)
        normal_matrix = mesh_obj["inverse_transform"][:3, :3].T
        obj_normal = normalize(normal_matrix @ local_normal)
    else:
        obj_normal = get_normal(hit_obj, intersect_point)  # pyright: ignore[reportArgumentType]
    dir_to_light = normalize(light - intersect_point)  # pyright: ignore[reportArgumentType]
    dir_to_origin = normalize(origin - intersect_point)  # pyright: ignore[reportArgumentType]

    # Shadow: any mesh hit strictly between light and point => shadowed
    dir_from_light = normalize(intersect_point - light)  # pyright: ignore[reportArgumentType]
    dist_light_to_intersect_point = np.linalg.norm(light - intersect_point)
    t_max_shadow = dist_light_to_intersect_point - 1e-4  # Avoid self-hit at the endpoint

    # == Early exit (compute)
    # Front-to-back traversal to stop at first valid hit.
    if EARLY_EXIT:
        best_hit_dist = np.inf
        hit_found = False

        # Unbounded / non-mesh objects are not in any tree: test directly.
        for candidate in scene:
            match candidate["type"]:
                case ObjectType.PLANE:
                    t_obj = intersect_plane(
                        light, dir_from_light, candidate["position"], candidate["normal"]
                    )
                    if 0 <= t_obj < best_hit_dist and t_obj < t_max_shadow:
                        best_hit_dist, hit_found = t_obj, True
                case ObjectType.SPHERE:
                    t_obj = intersect_sphere(
                        light, dir_from_light, candidate["position"], candidate["radius"]
                    )
                    if 0 <= t_obj < best_hit_dist and t_obj < t_max_shadow:
                        best_hit_dist, hit_found = t_obj, True

        # Pick traversal root; `root_aabb` mode has no tree, so use a single
        # "super node" holding every face (brute-force batch).
        if ACCELERATION == "octree" and octree is not None:
            root_node = octree.root
        elif ACCELERATION == "bvh" and bvh is not None:
            root_node = bvh.root
        else:
            root_node = None  # sentinel

        stack: list[tuple[float, OctreeNode | BVHNode | None]] = []
        root_interval = (
            intersect_aabb(light, dir_from_light, root_node.bounds) if root_node else (0.0, np.inf)
        )
        if root_interval:
            stack.append((root_interval[0], root_node))

        while stack:
            node_near, node = stack.pop()
            if node_near > t_max_shadow:
                continue

            # This node's faces, in one vectorized any-hit pass. A shadow ray
            # only needs to know whether an occluder exists.
            faces = all_mesh_faces(scene) if node is None else node.object_indices

            global triangle_intersection_test_count
            triangle_intersection_test_count += len(faces)

            candidates_by_mesh: dict[int, list[int]] = {}
            for mesh_index, face_index in faces:
                if (
                    hit_mesh_index is not None
                    and hit_face is not None
                    and (mesh_index, face_index) == (hit_mesh_index, hit_face)
                ):
                    continue
                candidates_by_mesh.setdefault(mesh_index, []).append(face_index)

            for mesh_index, face_indices in candidates_by_mesh.items():
                mesh_obj = scene[mesh_index]
                inverse_transform = mesh_obj["inverse_transform"]
                local_origin = transform_point(inverse_transform, light)
                local_direction = transform_vector(inverse_transform, dir_from_light)
                if any_triangle_hit_batch(
                    mesh_obj["mesh"],
                    local_origin,
                    local_direction,
                    face_indices,
                    t_max=t_max_shadow,
                ):
                    return None

            # Push the farther child first so the nearer child is visited
            # next without heap allocation.
            if node and node.children:
                children: list[tuple[float, OctreeNode | BVHNode]] = []
                for child in node.children:
                    child_interval = intersect_aabb(light, dir_from_light, child.bounds)
                    if child_interval is None:
                        continue
                    child_near, _ = child_interval
                    if child_near > t_max_shadow:
                        continue
                    children.append((child_near, child))
                children.sort(key=lambda item: item[0], reverse=True)
                stack.extend(children)

        if hit_found and best_hit_dist < dist_light_to_intersect_point:
            return None
    else:
        shadow_hits = intersect_scene(
            scene,
            root_bounds,
            light,
            dir_from_light,
            octree,
            bvh,
            t_max=t_max_shadow,
            exclude_face=(
                (hit_mesh_index, hit_face)
                if hit_mesh_index is not None and hit_face is not None
                else None
            ),
        )
        mesh_shadow = any(
            kind == "mesh" and payload[0] < dist_light_to_intersect_point
            for kind, payload in shadow_hits
        )
        other_shadow = any(
            kind != "mesh" and payload[0] < dist_light_to_intersect_point
            for kind, payload in shadow_hits
        )
        if mesh_shadow or other_shadow:
            return None

    # Shading -- Compute pixel color
    ray_color = glob_ambient
    obj_color = (
        scene[hit_mesh_index]["mesh"].colors[hit_face]
        if hit_mesh_index is not None and hit_face is not None
        else get_color(hit_obj, intersect_point)  # pyright: ignore[reportArgumentType]
    )
    diffuse = glob_diffuse_coeff
    ray_color += diffuse * max(np.dot(obj_normal, dir_to_light), 0) * obj_color
    # == Blinn-Phong model (baseline)
    ray_color += (
        glob_specular_coeff
        * max(np.dot(obj_normal, normalize(dir_to_light + dir_to_origin)), 0) ** glob_specular_k
    )
    # /== Blinn-Phong model
    reflection = (
        scene[hit_mesh_index]["reflection"]
        if hit_mesh_index is not None
        else hit_obj.get("reflection", 0.5)  # pyright: ignore[reportOptionalMemberAccess]
    )

    return intersect_point, obj_normal, ray_color, reflection


def add_sphere(pos: Point3, radius: float, color: RGB) -> Object:
    return {
        "type": ObjectType.SPHERE,
        "position": np.array(pos),
        "radius": np.float64(radius),
        "color": np.array(color),
        "reflection": 0.5,
    }


def get_plane_color(intersect_point: Point) -> Vector:
    # Inigo Quilez-style signed distance to a 1x1 rectangle in the XZ plane.
    xz = intersect_point[[0, 2]]
    q = np.abs(xz) - plane_sdf_half_size
    sdf = np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1]), 0.0)
    band = int(np.floor(sdf / plane_sdf_band_width))
    return plane_orange_palette[band % len(plane_orange_palette)]


def add_plane(pos: Point3, normal: Vec3) -> Object:
    return {
        "type": ObjectType.PLANE,
        "position": np.array(pos),
        "normal": np.array(normal),
        "color": get_plane_color,
        "diffuse_c": 0.75,
        "specular_c": 0.5,
        "reflection": 0.25,
    }


def add_mesh(mesh: Mesh, transform: np.ndarray | None = None) -> Object:
    if transform is None:
        transform = np.eye(4, dtype=np.float32)

    return {
        "type": ObjectType.MESH,
        "mesh": mesh,
        "transform": transform,
        "inverse_transform": np.linalg.inv(transform),
        "reflection": 0.5,
    }


def _reset_intersection_counters() -> None:
    global intersection_test_count
    global triangle_intersection_test_count
    global plane_intersection_test_count
    global sphere_intersection_test_count

    intersection_test_count = 0
    triangle_intersection_test_count = 0
    plane_intersection_test_count = 0
    sphere_intersection_test_count = 0
    aabb_module.aabb_intersection_test_count = 0


def _render_tile(
    tile: tuple[int, int, int, int],
    state: RenderState,
) -> tuple[int, int, np.ndarray, tuple[int, int, int, int, int]]:
    scene, light, root_bounds, octree, bvh, cam_origin, camera_ray_directions = state
    tile_x, tile_y, tile_x_end, tile_y_end = tile
    tile_image = np.zeros((tile_y_end - tile_y, tile_x_end - tile_x, 3))
    start_counts = (
        intersection_test_count,
        aabb_module.aabb_intersection_test_count,
        plane_intersection_test_count,
        sphere_intersection_test_count,
        triangle_intersection_test_count,
    )

    for local_y, ray_row in enumerate(camera_ray_directions[tile_y:tile_y_end, tile_x:tile_x_end]):
        for local_x, ray_direction in enumerate(ray_row):
            pixel_color = np.zeros(3, dtype=np.float32)
            ray_origin = cam_origin
            depth = 0
            reflection = 1.0
            while depth < max_depth:
                traced = trace_ray(
                    scene,
                    light,
                    ray_origin,
                    ray_direction,
                    root_bounds,
                    octree,
                    bvh,
                )
                if traced is None:
                    break
                intersect_point, obj_normal, ray_color, reflected = traced
                ray_origin = intersect_point + obj_normal * 1e-4
                ray_direction = normalize(
                    ray_direction - 2 * np.dot(ray_direction, obj_normal) * obj_normal
                )
                depth += 1
                pixel_color += reflection * ray_color
                reflection *= reflected

            tile_image[tile_y_end - tile_y - local_y - 1, local_x, :] = np.clip(pixel_color, 0, 1)

    return (
        tile_x,
        tile_y,
        tile_image,
        (
            intersection_test_count - start_counts[0],
            aabb_module.aabb_intersection_test_count - start_counts[1],
            plane_intersection_test_count - start_counts[2],
            sphere_intersection_test_count - start_counts[3],
            triangle_intersection_test_count - start_counts[4],
        ),
    )


def _initialize_render_worker(state: RenderState) -> None:
    global _worker_render_state
    _worker_render_state = state
    _reset_intersection_counters()


def _render_tile_worker(
    tile: tuple[int, int, int, int],
) -> tuple[int, int, np.ndarray, tuple[int, int, int, int, int]]:
    if _worker_render_state is None:
        raise RuntimeError("Render worker was not initialized.")
    return _render_tile(tile, _worker_render_state)


def main():
    total_start = perf_counter()
    global intersection_test_count
    global plane_intersection_test_count
    global sphere_intersection_test_count
    global triangle_intersection_test_count

    root = Path(__file__).parent
    model = f"{root}/meshes/bun_zipper.ply"

    load_start = perf_counter()
    vertices, faces = bunny.load(model, scale=5.0)
    # === Meme configuration (do not remove) ===
    # mirror_vertices = vertices.copy()
    # mirror_vertices[:, 0] = -mirror_vertices[:, 0]
    # sm_vertices, sm_faces = bunny.load(model, scale=4.0)
    # /== Meme configuration /==
    bunny_colors: Final = (
        (1.0, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.8, 0.2),
        (0.0, 0.4, 1.0),
        (0.2, 0.0, 0.8),
        (0.6, 0.0, 0.8),
    )
    rainbow_meshes = [build_mesh(vertices, faces, color=color) for color in bunny_colors]
    bunny_white = build_mesh(vertices, faces, color=(1.0, 1.0, 1.0))
    load_end = perf_counter()

    # === Populate scene ===
    rainbow_layout: Final = (
        (-2.4, 0.30, -0.15, np.pi / 10),
        (-1.6, 0.12, -0.05, np.pi / 18),
        (-0.8, 0.04, 0.05, -np.pi / 24),
        (0.8, 0.04, 0.05, np.pi / 24),
        (1.6, 0.12, -0.05, -np.pi / 18),
        (2.4, 0.30, -0.15, -np.pi / 10),
        (-0.2, 1.55, 0.15, np.pi),
    )
    letter_sphere_radius: Final = 0.12
    letter_sphere_z: Final = -1.5
    letter_sphere_color: Final = (1.0, 1.0, 1.0)

    def lerp(A, B, N: int, include=False) -> list[tuple[float, float]]:
        result = []
        for i in range(N):
            t = i / N
            x = A[0] + (B[0] - A[0]) * t
            y = A[1] + (B[1] - A[1]) * t
            result.append((x, y))
        result.append(B) if include else ...
        return result

    a_spheres: Final = (
        # Left
        *lerp((-1.05, 2.0), (-0.45, 3.3), 8, include=True),
        # Right
        *lerp((0.25, 2.0), (-0.45, 3.3), 8),
        # Middle
        *lerp((-0.6, 2.55), (-0.0, 2.55), 3),
    )
    plus_spheres: Final = (
        # Vertical
        (0.75, 3.0),
        (0.75, 3.25),
        (0.75, 3.5),
        (0.75, 3.75),
        (0.75, 4.0),
        # Horizontal
        (0.25, 3.5),
        (0.5, 3.5),
        (0.75, 3.5),
        (1.0, 3.5),
        (1.25, 3.5),
    )
    scene: Final = [
        add_plane((0.0, 0.15, 0.0), (0.0, 1.0, 0.0)),
        *(
            add_mesh(
                rainbow_meshes[index],
                translation(x, y, z) @ rotation_z(angle),
            )
            for index, (x, y, z, angle) in enumerate(rainbow_layout)
        ),
        add_mesh(bunny_white),
        # --- A+ ---
        # *(
        #     add_sphere((x, y, letter_sphere_z), letter_sphere_radius, letter_sphere_color)
        #     for x, y in (*a_spheres, *plus_spheres)
        # ),
        # --- Two Spheres ---
        # add_sphere((1.25, 1 + 0.15, -5), 1.0, (0 / 255, 255 / 255, 255 / 255)),
        # add_sphere((-1.25, 1 + 0.15, -5), 1.0, (255 / 255, 128 / 255, 128 / 255)),
        # === Meme configuration (Do not remove) ===
        # add_mesh(bunny_white),
        # add_mesh(bunny_black, translation(0.1, 0.0, -1.0) @ rotation_y(np.pi * 1 / 6)),
        # add_mesh(bunny_black, translation(1.0, 0.0, -1.0)),
        # add_mesh(bunny_black, translation(2.0, 0.0, -1.0) @ rotation_y(-np.pi * 1 / 12)),
        # add_mesh(bunny_black_mirror, translation(-1.0, 0.0, -1.0)),
        # add_mesh(bunny_black_mirror, translation(-2.0, 0.0, -1.0) @ rotation_y(np.pi * 1 / 12)),
        # /== Meme configuration /==
    ]

    precompute_start = perf_counter()
    # == Root AABB precomputation (compute)
    # The finite scene bounds are computed once instead of per-camera and
    # per-ray. The plane is intentionally excluded because it is unbounded.
    # scene_root_aabb = root_aabb(scene)
    scene_root_aabb = None
    # == Octree precomputation (compute)
    # Build the alternative hierarchy once before rendering so its construction
    # cost is not repeated for camera or shadow rays.
    # scene_octree = build_octree(scene)
    scene_octree = None
    # == BVH precomputation (compute)
    # Build the alternative hierarchy once before rendering so its construction
    # cost is not repeated for camera or shadow rays.
    build_bvh_start = perf_counter()
    scene_bvh = build_bvh(scene, max_objects=32)
    build_bvh_end = perf_counter()

    # === Configure camera/viewpoint ===
    cam_origin: Final = np.array([-0.1, 1.0, 5], dtype=np.float32)

    image = np.zeros((HEIGHT, WIDTH, 3))
    aspect_ratio = float(WIDTH) / HEIGHT
    screen = (-3.5, -3.5 / aspect_ratio + 0.25, 3.5, 3.5 / aspect_ratio + 0.25)

    # Precompute all primary camera rays once. Tile traversal below reuses
    # these directions without rebuilding and normalizing one ray per pixel.
    screen_x = np.linspace(screen[0], screen[2], WIDTH, dtype=np.float32)
    screen_y = np.linspace(screen[1], screen[3], HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(screen_x, screen_y)
    camera_targets = np.stack(
        (grid_x, grid_y, np.zeros_like(grid_x)),
        axis=-1,
    )
    camera_ray_directions = camera_targets - cam_origin
    camera_ray_directions /= np.linalg.norm(camera_ray_directions, axis=-1, keepdims=True)
    precompute_end = perf_counter()

    state: RenderState = (
        scene,
        light_pos,
        scene_root_aabb,
        scene_octree,
        scene_bvh,
        cam_origin,
        camera_ray_directions,
    )
    tiles = [
        (tile_x, tile_y, min(tile_x + TILE_SIZE, WIDTH), min(tile_y + TILE_SIZE, HEIGHT))
        for tile_y in range(0, HEIGHT, TILE_SIZE)
        for tile_x in range(0, WIDTH, TILE_SIZE)
    ]

    _reset_intersection_counters()
    render_start = perf_counter()
    if MULTIPROC:
        with ProcessPoolExecutor(
            initializer=_initialize_render_worker,
            initargs=(state,),
        ) as executor:
            results = executor.map(_render_tile_worker, tiles)
            for tile_number, result in enumerate(results, start=1):
                tile_x, tile_y, tile_image, counts = result
                tile_height, tile_width = tile_image.shape[:2]
                image[
                    HEIGHT - tile_y - tile_height : HEIGHT - tile_y,
                    tile_x : tile_x + tile_width,
                ] = tile_image
                (
                    intersection_delta,
                    aabb_delta,
                    plane_delta,
                    sphere_delta,
                    triangle_delta,
                ) = counts
                intersection_test_count += intersection_delta
                aabb_module.aabb_intersection_test_count += aabb_delta
                plane_intersection_test_count += plane_delta
                sphere_intersection_test_count += sphere_delta
                triangle_intersection_test_count += triangle_delta
                if tile_number == 1 or tile_number % max(1, len(tiles) // 10) == 0:
                    print(
                        f"\r\x1b[0J{100 * tile_number / len(tiles):.2f}%",
                        end="",
                        flush=True,
                    )
    else:
        for tile_number, tile in enumerate(tiles, start=1):
            tile_x, tile_y, _, _ = tile
            _, _, tile_image, _ = _render_tile(tile, state)
            tile_height, tile_width = tile_image.shape[:2]
            image[
                HEIGHT - tile_y - tile_height : HEIGHT - tile_y,
                tile_x : tile_x + tile_width,
            ] = tile_image
            if tile_number == 1 or tile_number % max(1, len(tiles) // 10) == 0:
                print(
                    f"\r\x1b[0J{100 * tile_number / len(tiles):.2f}%",
                    end="",
                    flush=True,
                )

    render_end = perf_counter()
    total_end = perf_counter()
    print("\r\x1b[0JDone.")

    def tests_per_pixel(count: int, out: type = str, width=WIDTH, height=HEIGHT) -> float | str:
        result = count / (width * height)
        if out == str:
            return f"{result:.2f} / px"
        return out(result)

    def tests_per_faces(count: int, scene: list[Object], out: type = str) -> float | str:
        meshes = [obj["mesh"] for obj in scene if obj["type"] == ObjectType.MESH]
        if not meshes:
            result = 0
        else:
            result = count / sum(len(mesh.faces) for mesh in meshes)  # pyright: ignore[reportAttributeAccessIssue]
        if out == str:
            return f"{result:.2f} / face"
        return out(result)

    def get_bunny_count() -> int:
        count = 0
        for obj in scene:
            if obj["type"] == ObjectType.MESH:
                count += 1
        return count

    # Band-Aid to get updated count
    global aabb_intersection_test_count
    aabb_intersection_test_count += 0

    print("=== Load Time ===")
    print(f"Bunny model: {ANSI.GREEN}{model.split('/')[-1]}{ANSI.RESET}")
    print(f"Model count: {fmt.color(ANSI.BRIGHT_MAGENTA, get_bunny_count())}")
    print(f"Load 3D model(s): {(load_end - load_start):.3f} s")
    print("=== Precompute Time ===")
    print(f"Build BVH: {(build_bvh_end - build_bvh_start):.3f} s")
    print(
        f"Precompute (total): {ANSI.YELLOW}{(precompute_end - precompute_start):.3f}{ANSI.RESET} s"
    )
    print("=== Render Time ===")
    print("Configuration:")
    print(f"\tShading model: {ANSI.BRIGHT_BLUE}{SHADING}{ANSI.RESET}")
    print(f"\tIntersection tests optimization method: {ANSI.BRIGHT_BLUE}{ACCELERATION}{ANSI.RESET}")
    print(f"\tCulling: {fmt.pprint_bool(CULLING)}")
    print(f"\tEarly exit: {fmt.pprint_bool(EARLY_EXIT)}")
    print(f"\tVectorized Möller-Trumbore: {fmt.pprint_bool(VECTORIZED)}")
    print(f"\tPrimary-ray tile size: {TILE_SIZE}")
    print(f"\tMultiprocessing: {fmt.pprint_bool(MULTIPROC)}")
    fmt.newline()
    print(f"Render: {ANSI.YELLOW}{(render_end - render_start):.3f}{ANSI.RESET} s")
    print("Intersection tests:")
    aabb_intersection_test_count = pprint_aabb_tests(WIDTH, HEIGHT)
    print(
        f"\tTriangle: {triangle_intersection_test_count} "
        f"({fmt.color(ANSI.DIM, tests_per_pixel(triangle_intersection_test_count))}, "
        f"{fmt.color(ANSI.DIM, tests_per_faces(triangle_intersection_test_count, scene))})"
    )
    print(
        f"\tPlane: {plane_intersection_test_count} "
        f"({fmt.color(ANSI.DIM, tests_per_pixel(plane_intersection_test_count))})"
    )
    if sphere_intersection_test_count > 0:
        print(
            f"\tSphere: {sphere_intersection_test_count} "
            f"({fmt.color(ANSI.DIM, tests_per_pixel(plane_intersection_test_count))})"
        )
    total_intersection_test_count = (
        intersection_test_count
        + aabb_intersection_test_count
        + plane_intersection_test_count
        + triangle_intersection_test_count
        + sphere_intersection_test_count
    )
    print(
        f"\t{ANSI.UNDERLINE}{fmt.bold('Total')}{ANSI.RESET}: "
        f"{fmt.color(ANSI.CYAN, total_intersection_test_count)} "
        f"({fmt.color(ANSI.DIM, tests_per_pixel(total_intersection_test_count))})"
    )
    print(
        f"{fmt.bold(f'{ANSI.UNDERLINE}Total time')}: {ANSI.YELLOW}{(total_end - total_start):.3f}{ANSI.RESET} s"
    )
    print("-" * 40)
    Image.fromarray((255 * image).astype(np.uint8), "RGB").save("out.png")


if __name__ == "__main__":
    main()
