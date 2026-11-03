"""
Re-implementation of `raytracing.py` from the provided codebase.
"""

import random
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Final

import numpy as np
from PIL import Image

from gro821.problematique2 import bunny
from gro821.problematique2.aabb import AABB, intersects_aabb, root_aabb
from gro821.problematique2.bvh import BVH, build_bvh, query_bvh
from gro821.problematique2.octree import Octree, build_octree, query_octree

type Vector = np.ndarray[tuple[int], np.dtype[np.float32]]
Point = Vertex = Vector

type Vec3 = tuple[float, float, float]
Point3 = Vec3
type RGB = tuple[float, float, float]


class ObjectType(StrEnum):
    PLANE = "plane"
    SPHERE = "sphere"
    TRIANGLE = "triangle"


type Object = dict[str, Any]

# Global parameters
plane_color_pair: Final = (
    np.array([0, 0, 0], dtype=np.float32),
    np.array([1, 1, 1], dtype=np.float32),
)
light_pos: Final = np.array([5, 5, -10], dtype=np.float32)
light_color = np.array([1, 1, 1], dtype=np.float32)

glob_ambient: Final = 0.05
glob_diffuse_coeff: Final = 1.0
glob_specular_coeff: Final = 1.0
glob_specular_k: Final = 50

max_depth: Final = 1  # Upper limit on number of reflections

WIDTH = 200
HEIGHT = 150

SHADING = "flat"  # "flat" | "gouraud"

# Ray rejection method
ACCELERATION = "bvh"  # "root_aabb" | "octree" | "bvh"

# == Intersection test count benchmark
intersection_test_count = 0


def normalize(x: Vector):
    x /= np.linalg.norm(x)
    return x


def intersect_plane(origin: Point, direction: Vector, plane: Point, normal: Vector) -> float:
    den = np.dot(direction, normal)
    if np.abs(den) <= 1e-6:
        return np.inf
    distance = np.dot(plane - origin, normal) / den
    return float(distance) if distance >= 0 else np.inf


def intersect_sphere(origin: Point, direction: Vector, sphere: Point, radius: float) -> float:
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
    EPS: Final = 1e-6
    # == Möller-Trumbore without redundant work (compute)
    # Optimized version of Möller-Trumbore.
    e1, e2 = triangle["e1"], triangle["e2"]
    p1 = triangle["p1"]

    # Use the standard Möller-Trumbore edge order
    pv = np.cross(direction, e2)
    det = np.dot(e1, pv)
    if abs(det) < EPS:
        return np.inf, 0.0, 0.0
    # == Back-face culling (compute)
    # Essentially a "win for free".
    # NOTE: did NOT reduce the number of tests.
    if det >= 0:
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


def intersect_scene(
    scene: list[Object],
    root_bounds: AABB | None,
    origin: Point,
    direction: Vector,
    octree: Octree | None = None,
    bvh: BVH | None = None,
    t_max: float = np.inf,
) -> list[tuple[int, tuple[float, float, float]]]:
    # == Selectable spatial acceleration (compute)
    # Keep the root AABB baseline available for comparison while allowing the
    # other methods to reject objects using smaller node bounds.
    if ACCELERATION == "octree":
        finite_object_indices = query_octree(octree, origin, direction, t_max)
    elif ACCELERATION == "bvh":
        finite_object_indices = query_bvh(bvh, origin, direction, t_max)
    elif ACCELERATION == "root_aabb":
        finite_objects_visible = root_bounds is None or intersects_aabb(
            origin, direction, root_bounds
        )
        finite_object_indices = (
            [i for i, obj in enumerate(scene) if obj["type"] != ObjectType.PLANE]
            if finite_objects_visible
            else []
        )
    else:
        raise ValueError(f"Unsupported acceleration mode: {ACCELERATION}.")

    intersections = [(i, intersect(origin, direction, scene[i])) for i in finite_object_indices]
    intersections.extend(
        (i, intersect(origin, direction, obj))
        for i, obj in enumerate(scene)
        if obj["type"] == ObjectType.PLANE
    )
    return intersections


def get_normal(obj: Object, intersect_point: Point, u: float = 0.0, v: float = 0.0) -> Vector:
    """Compute the normal of an object."""
    match obj["type"]:
        case ObjectType.PLANE:
            # == Use the plane's geometric normal for shading
            plane_normal = np.asarray(obj["normal"], dtype=np.float32)
            return plane_normal / np.linalg.norm(plane_normal)
        case ObjectType.SPHERE:
            return normalize(intersect_point - obj["position"])
        case ObjectType.TRIANGLE:
            if SHADING == "flat":
                return obj["normal"]
            elif SHADING == "gouraud":
                # == Gouraud shading -- Barycentric interpolation
                f = FACES[obj["face_idx"]]
                normal = (
                    (1 - u - v) * VERTICES_NORMALS[f[0]]
                    + u * VERTICES_NORMALS[f[1]]
                    + v * VERTICES_NORMALS[f[2]]
                )
                return normalize(normal)  # pyright: ignore[reportArgumentType]
        case _:
            raise ValueError(f"Unsupported object type: {obj['type']}.")
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
) -> tuple[Object, Point, Vector, Vector] | None:
    t = np.inf
    u = v = 0.0
    obj_index: int = len(scene) + 1  # OOB initalization
    for obj_index_candidate, intersection in intersect_scene(
        scene, root_bounds, origin, direction, octree, bvh
    ):
        obj_intersect_dist, u_hit, v_hit = intersection
        if obj_intersect_dist < t:
            t, obj_index, u, v = obj_intersect_dist, obj_index_candidate, u_hit, v_hit

    if t == np.inf:
        return None
    obj = scene[obj_index]

    # Find point of insertect on object
    intersect_point = origin + direction * t
    obj_normal = get_normal(obj, intersect_point, u, v)  # pyright: ignore[reportArgumentType]
    obj_color = get_color(obj, intersect_point)  # pyright: ignore[reportArgumentType]
    dir_to_light = normalize(light - intersect_point)  # pyright: ignore[reportArgumentType]
    dir_to_origin = normalize(origin - intersect_point)  # pyright: ignore[reportArgumentType]

    # Find if point is shadowed
    dir_from_light = normalize(intersect_point - light)  # pyright: ignore[reportArgumentType]
    # == Distance pruning (compute)
    dist_light_to_point = np.linalg.norm(light - intersect_point)
    t_max_shadow = dist_light_to_point - 1e-4  # Avoid self-hit at the endpoint
    intersects = [
        intersection[0]
        for candidate_index, intersection in intersect_scene(
            scene, root_bounds, light, dir_from_light, octree, bvh, t_max=t_max_shadow
        )
        if scene[candidate_index] is not obj
    ]
    dist_light_to_intersect_point = np.linalg.norm(light - intersect_point)
    # == Distance pruning
    # Distance check is now implicit in t_max, but kept as a cheap safety net
    # (e.g. for "root_aabb" mode, which ignores t_max).
    if intersects and min(intersects) < dist_light_to_intersect_point:
        return None

    # Compute pixel color
    global glob_ambient, glob_diffuse_coeff, glob_specular_coeff, glob_specular_k  # noqa: PLW0602
    ray_color = glob_ambient

    diffuse_coeff = obj.get("diffuse_c", glob_diffuse_coeff)
    assert isinstance(diffuse_coeff, float)
    ray_color += diffuse_coeff * max(np.dot(obj_normal, dir_to_light), 0) * obj_color

    specular_coeff = obj.get("specular_c", glob_specular_coeff)
    assert isinstance(specular_coeff, float)
    ray_color += (
        specular_coeff
        * max(np.dot(obj_normal, normalize(dir_to_light + dir_to_origin)), 0) ** glob_specular_k
    )

    return obj, intersect_point, obj_normal, ray_color  # pyright: ignore[reportReturnType]


def add_sphere(pos: Point3, radius: float, color: RGB) -> Object:
    return {
        "type": ObjectType.SPHERE,
        "position": np.array(pos),
        "radius": np.float64(radius),
        "color": np.array(color),
        "reflection": 0.5,
    }


def get_plane_color(intersect_point: Point) -> Vector:
    x_index = int(intersect_point[0] * 2) % 2
    z_index = int(intersect_point[2] * 2) % 2
    return plane_color_pair[x_index == z_index]


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


def add_triangle(face_idx: int, color: RGB) -> Object:
    # == Precomputed triangle properties and quantities (memory, compute)
    # Scene triangles stop storing `np.array` copies of their corners; they
    # store indices or a slice/view into shared arrays, so every triangle
    # references the same vertex store (mesh connectivity preserved; duplicated
    # corner data disappears). Precomputed edges and flat normals at laod time
    # saves per-ray operations at render time.
    return {
        "type": ObjectType.TRIANGLE,
        "face_idx": face_idx,
        "e1": EDGES_1[face_idx],  # pyright: ignore[reportUnboundVariable]
        "e2": EDGES_2[face_idx],  # pyright: ignore[reportUnboundVariable]
        "normal": NORMALS[face_idx],  # pyright: ignore[reportUnboundVariable]
        "p1": VERTICES[FACES[face_idx, 0]],  # pyright: ignore[reportUnboundVariable]
        "color": np.array(color),
        "reflection": 0.5,
    }


def main():
    root = Path(__file__).parent
    global VERTICES, EDGES_1, EDGES_2, FACES, NORMALS  # Triangles quantities (flat-shading)
    global VERTICES_NORMALS  # Vertices quantities (Gouraud shading)

    load_start = perf_counter()
    VERTICES, FACES = bunny.load(f"{root}/meshes/bun_zipper_res3.ply", scale=5.0)
    load_end = perf_counter()

    # == Precomputation of triangle properties (compute)
    # Moves per-triangle precomputation out of the render loop. The entire
    # flat-shading precomputation collapses into a few lines of vectorized
    # operations, done once, instead of per-ray Python arithmetic. Then,
    # `intersect_triangles` receives `edge1, edge2` instead of precomputing
    # `b - a` and `c - a` billions of times.
    flat_shading_precompute_start = perf_counter()
    p1 = VERTICES[FACES[:, 0]]  # (M, 3)
    p2 = VERTICES[FACES[:, 1]]
    p3 = VERTICES[FACES[:, 2]]

    EDGES_1 = p2 - p1  # (M, 3)
    EDGES_2 = p3 - p1  # (M, 3)
    NORMALS = normalize(np.cross(EDGES_1, EDGES_2))  # Face normals
    flat_shading_precompute_end = perf_counter()

    # == Gouraud shading -- Precompute vertex normals
    gouraud_shading_precompute_start = perf_counter()
    VERTICES_NORMALS = np.zeros_like(VERTICES)  # (N, 3), one row per vertex
    # Each face contributes its normal to its 3 vertices
    for col in (0, 1, 2):
        np.add.at(VERTICES_NORMALS, FACES[:, col], NORMALS)

    # Normalize each row (avoid DivByZero for unreferenced vertices)
    lengths = np.linalg.norm(VERTICES_NORMALS, axis=1, keepdims=True)
    np.divide(VERTICES_NORMALS, lengths, out=VERTICES_NORMALS, where=lengths > 0)
    gouraud_shading_precompute_end = perf_counter()

    # === Populate scene ===
    scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
    # scene.append(add_sphere((1, 1, 10), 1.0, (0.8, 0.6, 1.0)))
    # for i in random.sample(range(len(FACES)), len(FACES) // 10):
    #     scene.append(add_triangle(i, (255.0 * 0.7, 255.0 * 0.7, 255.0 * 0.7)))
    for i in range(len(FACES)):
        scene.append(add_triangle(i, (255.0 * 0.7, 255.0 * 0.7, 255.0 * 0.7)))

    # == Root AABB precomputation (compute)
    # The finite scene bounds are computed once instead of per-camera and
    # per-ray. The plane is intentionally excluded because it is unbounded.
    scene_root_aabb = root_aabb(scene)
    # == Octree precomputation (compute-wise)
    # Build the alternative hierarchy once before rendering so its construction
    # cost is not repeated for camera or shadow rays.
    scene_octree = build_octree(scene)
    # == BVH precomputation (compute-wise)
    # Build the alternative hierarchy once before rendering so its construction
    # cost is not repeated for camera or shadow rays.
    scene_bvh = build_bvh(scene)

    # === Configure camera/viewpoint ===
    pixel_color = np.array([0, 0, 0], dtype=np.float64)
    cam_origin: Final = np.array([0, 0.35, -3], dtype=np.float64)
    cam_focal_point: Final = np.array([0, 0, 1], dtype=np.float64)

    image = np.zeros((HEIGHT, WIDTH, 3))
    aspect_ratio = float(WIDTH) / HEIGHT
    screen = (-1.0, -1.0 / aspect_ratio + 0.25, 1.0, 1.0 / aspect_ratio + 0.25)

    global intersection_test_count
    intersection_test_count = 0
    render_start = perf_counter()
    for i, x in enumerate(np.linspace(screen[0], screen[2], WIDTH)):
        if i % 10 == 0:
            print(f"\r\x1b[0J{100 * i / WIDTH}%", end="")
        for j, y in enumerate(np.linspace(screen[1], screen[3], HEIGHT)):
            pixel_color[:] = 0.0
            cam_focal_point[:2] = (x, y)
            ray_origin = cam_origin
            ray_direction = normalize(cam_focal_point - cam_origin)  # pyright: ignore[reportArgumentType]
            depth: int = 0
            reflection = 1.0
            while depth < max_depth:
                traced = trace_ray(
                    scene,
                    light_pos,
                    ray_origin,  # pyright: ignore[reportArgumentType]
                    ray_direction,
                    scene_root_aabb,
                    scene_octree,
                    scene_bvh,
                )  # pyright: ignore[reportArgumentType]
                if traced is None:
                    break
                obj, intersect_point, obj_normal, ray_color = traced
                ray_origin = intersect_point + obj_normal * 1e-4
                ray_direction = normalize(
                    ray_direction - 2 * np.dot(ray_direction, obj_normal) * obj_normal  # pyright: ignore[reportArgumentType]
                )
                depth += 1
                pixel_color += reflection * ray_color
                obj_reflection = obj.get("reflection", 1.0)
                assert isinstance(obj_reflection, float)
                reflection *= obj_reflection

            image[HEIGHT - j - 1, i, :] = np.clip(pixel_color, 0, 1)

    render_end = perf_counter()
    print("\r\x1b[0JDone.")

    print("=== Load Time ===")
    print(f"Load 3D model: {(load_end - load_start):.3f}s")
    print(
        f"Flat-shading precompute: {(flat_shading_precompute_end - flat_shading_precompute_start):.3f}s"
    )
    print(
        f"Gouraud-shading precompute: {(gouraud_shading_precompute_end - gouraud_shading_precompute_start):.3f}s"
    )
    print("=== Render Time ===")
    print(f"Shading model: {SHADING}")
    print(f"Intersection tests optimization method: {ACCELERATION}")
    print(f"Render: {(render_end - render_start):.3f} s")
    print(
        f"Intersection tests: {intersection_test_count} ({intersection_test_count / (WIDTH * HEIGHT):.2f} / px)"
    )
    Image.fromarray((255 * image).astype(np.uint8), "RGB").save("out.png")
