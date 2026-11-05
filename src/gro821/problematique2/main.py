"""
Re-implementation of `raytracing.py` from the provided codebase.
"""

import heapq
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Final

import numpy as np
from PIL import Image

from gro821.problematique2 import *
from gro821.problematique2 import bunny
from gro821.problematique2.aabb import AABB, intersect_aabb, pprint_aabb_tests, root_aabb
from gro821.problematique2.bvh import BVH, BVHNode, build_bvh, query_bvh
from gro821.problematique2.mesh import Mesh, build_mesh
from gro821.problematique2.octree import Octree, OctreeNode, build_octree, query_octree
from gro821.sandbox import fmt
from gro821.sandbox.fmt import ANSI

type Vector = np.ndarray[tuple[int], np.dtype[np.float32]]
Point = Vertex = Vector

type Vec3 = tuple[float, float, float]
Point3 = Vec3
type RGB = tuple[float, float, float]

type Intersection = tuple[int, tuple[float, float, float]]
type Hit = tuple[str, Any]  # ("mesh", (t, u, v, face_index)) | ("plane", t) | ...


class ObjectType(StrEnum):
    PLANE = "plane"
    SPHERE = "sphere"
    TRIANGLE = "triangle"
    MESH = "mesh"


type Object = dict[str, Any]

# Global parameters
plane_color_pair: Final = (
    np.array([0, 0, 0], dtype=np.float32),
    np.array([1, 1, 1], dtype=np.float32),
)
light_pos: Final = np.array([5, 5, 10], dtype=np.float32)
light_color = np.array([1, 1, 1], dtype=np.float32)

glob_ambient: Final = 0.05
glob_diffuse_coeff: Final = 1.0
glob_specular_coeff: Final = 1.0
glob_specular_k: Final = 50

max_depth: Final = 1  # Upper limit on number of reflections

WIDTH: Final = 200
HEIGHT: Final = 150

SHADING = "flat"  # "flat" | "gouraud"

# Ray rejection method
ACCELERATION = "bvh"  # "root_aabb" | "octree" | "bvh"

CULLING: Final = True
EARLY_EXIT: Final = True
VECTORIZED: Final = True


def normalize(x: Vector):
    x /= np.linalg.norm(x)
    return x


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
    not_culled = np.abs(det) >= EPS if CULLING else det < -EPS
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
        if abs(det) < EPS:
            return None
    else:
        if det >= -EPS:
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
    exclude_face: int | None = None,
) -> list[Hit] | list[Intersection]:
    mesh = next(obj["mesh"] for obj in scene if obj["type"] == ObjectType.MESH)

    # == Selectable spatial acceleration (test)
    # Keep the root AABB baseline available for comparison while allowing the
    # other methods to reject objects using smaller node bounds.
    if ACCELERATION == "octree":
        candidate_indices = query_octree(octree, origin, direction, t_max)
    elif ACCELERATION == "bvh":
        candidate_indices = query_bvh(bvh, origin, direction, t_max)
    elif ACCELERATION == "root_aabb":
        visible = root_bounds is None or intersect_aabb(origin, direction, root_bounds) is not None
        candidate_indices = list(range(len(mesh.faces))) if visible else []
    else:
        raise ValueError(f"Unsupported acceleration mode: {ACCELERATION}.")

    hits: list[Hit] = []

    if VECTORIZED:
        # One vectorized pass over all mesh candidates
        global triangle_intersection_test_count
        triangle_intersection_test_count += len(candidate_indices)
        best = intersect_triangles_batch(
            mesh, origin, direction, candidate_indices, t_max, exclude_face
        )
        if best is not None:
            hits.append(("mesh", best))
    else:
        # Scalar path: same tests, one face at a time.
        for face_index in candidate_indices:
            if face_index == exclude_face:
                continue
            result = intersect_triangle_mesh(mesh, origin, direction, face_index, t_max)
            if result:
                hits.append(("mesh", (*result, face_index)))

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
    mesh = next(obj["mesh"] for obj in scene if obj["type"] == ObjectType.MESH)

    t = np.inf
    u = v = 0.0
    hit_face: int | None = None
    hit_obj: Object | None = None

    for kind, payload in intersect_scene(scene, root_bounds, origin, direction, octree, bvh):
        if kind == "mesh":
            t_hit, u_hit, v_hit, face_hit = payload  # pyright: ignore[reportAssignmentType]
        else:
            t_hit = payload[0]
        if t_hit < t:
            t = t_hit
            if kind == "mesh":
                u, v, hit_face = u_hit, v_hit, face_hit  # pyright: ignore[reportPossiblyUnboundVariable]
                hit_obj = None
            else:
                hit_obj = payload[1]  # pyright: ignore[reportAssignmentType]
                hit_face = None

    if t == np.inf:
        return None

    # Find point of insersect on object
    intersect_point = origin + direction * t
    obj_normal = (
        mesh_normal(mesh, hit_face, u, v)
        if hit_face is not None
        else get_normal(hit_obj, intersect_point)  # pyright: ignore[reportArgumentType]
    )
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

        q: list[tuple[float, int, OctreeNode | BVHNode | None]] = []
        seq = 0  # tie-break counter
        root_interval = (
            intersect_aabb(light, dir_from_light, root_node.bounds) if root_node else (0.0, np.inf)
        )
        if root_interval:
            heapq.heappush(q, (root_interval[0], seq, root_node))
            seq += 1

        while q:
            node_near, _, node = heapq.heappop(q)
            # No later node can contain a closer or relevant hit
            if node_near >= best_hit_dist or node_near > t_max_shadow:
                break

            # This node's faces, in one vectorized pass. The batch returns
            # the closest hit within t_max, so best_hit_dist feeds straight
            # back in as the cutoff -- pruning inside the arithmetic too.
            faces = list(range(len(mesh.faces))) if node is None else node.object_indices

            global triangle_intersection_test_count
            triangle_intersection_test_count += len(faces)

            best = intersect_triangles_batch(
                mesh,
                light,
                dir_from_light,
                faces,
                t_max=min(best_hit_dist, t_max_shadow),
                exclude_face=hit_face,
            )
            if best:
                best_hit_dist, hit_found = best[0], True

            # Push children in ray-entry order (heap does the ordering)
            if node and node.children:
                for child in node.children:
                    child_interval = intersect_aabb(light, dir_from_light, child.bounds)
                    if child_interval is None:
                        continue
                    child_near, _ = child_interval
                    if child_near > t_max_shadow or child_near >= best_hit_dist:
                        continue
                    heapq.heappush(q, (child_near, seq, child))
                    seq += 1

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
            exclude_face=hit_face,
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
        mesh.colors[hit_face] if hit_face is not None else get_color(hit_obj, intersect_point)  # pyright: ignore[reportArgumentType]
    )
    diffuse = glob_diffuse_coeff
    ray_color += diffuse * max(np.dot(obj_normal, dir_to_light), 0) * obj_color
    ray_color += (
        glob_specular_coeff
        * max(np.dot(obj_normal, normalize(dir_to_light + dir_to_origin)), 0) ** glob_specular_k
    )
    reflection = (
        mesh.reflection[hit_face] if hit_face is not None else hit_obj.get("reflection", 0.5)  # pyright: ignore[reportOptionalMemberAccess]
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
    x_index = int(intersect_point[0] * 2) % 2
    z_index = int(intersect_point[2] * 2) % 2
    return plane_color_pair[x_index != z_index]


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


def add_mesh(mesh: Mesh) -> Object:
    return {"type": ObjectType.MESH, "mesh": mesh, "reflection": 0.5}


def main():
    root = Path(__file__).parent
    model = f"{root}/meshes/bun_zipper.ply"

    load_start = perf_counter()
    vertices, faces = bunny.load(model, scale=5.0)
    _bunny = build_mesh(vertices, faces, color=(230/255, 125/255, 246/255))
    load_end = perf_counter()

    # === Populate scene ===
    scene = [
        add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0)),
        add_mesh(_bunny),
    ]

    # == Root AABB precomputation (compute)
    # The finite scene bounds are computed once instead of per-camera and
    # per-ray. The plane is intentionally excluded because it is unbounded.
    scene_root_aabb = root_aabb(scene)
    # == Octree precomputation (compute)
    # Build the alternative hierarchy once before rendering so its construction
    # cost is not repeated for camera or shadow rays.
    scene_octree = build_octree(scene)
    # == BVH precomputation (compute)
    # Build the alternative hierarchy once before rendering so its construction
    # cost is not repeated for camera or shadow rays.
    scene_bvh = build_bvh(scene)

    # === Configure camera/viewpoint ===
    pixel_color = np.array([0, 0, 0], dtype=np.float32)
    cam_origin: Final = np.array([0, 0.35, 3], dtype=np.float32)
    cam_focal_point: Final = np.array([0, 0, 1], dtype=np.float32)

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
                intersect_point, obj_normal, ray_color, reflected = traced
                ray_origin = intersect_point + obj_normal * 1e-4
                ray_direction = normalize(
                    ray_direction - 2 * np.dot(ray_direction, obj_normal) * obj_normal  # pyright: ignore[reportArgumentType]
                )
                depth += 1
                pixel_color += reflection * ray_color
                reflection *= reflected

            image[HEIGHT - j - 1, i, :] = np.clip(pixel_color, 0, 1)

    render_end = perf_counter()
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

    # Band-Aid to get updated count
    global aabb_intersection_test_count
    aabb_intersection_test_count += 0

    print("=== Load Time ===")
    print(f"Bunny model: {ANSI.GREEN}{model.split('/')[-1]}{ANSI.RESET}")
    print(f"Load 3D model: {(load_end - load_start):.3f}s")
    print("=== Render Time ===")
    print("Configuration:")
    print(f"\tShading model: {ANSI.BRIGHT_BLUE}{SHADING}{ANSI.RESET}")
    print(f"\tIntersection tests optimization method: {ANSI.BRIGHT_BLUE}{ACCELERATION}{ANSI.RESET}")
    print(f"\tCulling: {fmt.pprint_bool(CULLING)}")
    print(f"\tEarly exit: {fmt.pprint_bool(EARLY_EXIT)}")
    print(f"\tVectorized Möller-Trumbore: {fmt.pprint_bool(VECTORIZED)}")
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
    print("-" * 40)
    Image.fromarray((255 * image).astype(np.uint8), "RGB").save("out.png")
