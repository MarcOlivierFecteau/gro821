from dataclasses import dataclass

import numpy as np

type Array = np.typing.NDArray[np.float32]
type Indices = np.typing.NDArray[np.int32]


@dataclass(slots=True)
class Mesh:
    vertices: Array  # (N, 3)
    faces: Indices  # (M, 3)
    edge1: Array  # (M, 3)
    edge2: Array  # (M, 3)
    p1: Array  # (M, 3)
    normals: Array  # (M, 3)
    bounds_min: Array  # (M, 3)
    bounds_max: Array  # (M, 3)
    vertex_normals: Array  # (N, 3)
    # Material quantities
    colors: Array  # (M, 3) -- per-face RGB
    reflection: Array  # (M,) -- per-face reflectivity


def build_mesh(vertices: Array, faces: Indices, color: tuple[float, float, float]) -> Mesh:
    # == Precomputation of triangle quantities (compute)
    # Moves per-triangle precomputation out of the render loop. The entire
    # flat-shading precomputation collapses into a few lines of vectorized
    # operations, done once, instead of per-ray Python arithmetic. Then,
    # `intersect_triangle` receives `edge1, edge2` instead of precomputing
    # `b - a` and `c - a` billions of times.
    p1 = vertices[faces[:, 0]]
    p2 = vertices[faces[:, 1]]
    p3 = vertices[faces[:, 2]]

    edge1 = p2 - p1
    edge2 = p3 - p1

    normals = np.cross(edge1, edge2)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # == Gouraud shading -- Precompute vertex normals
    vertex_normals = np.zeros_like(vertices)
    for col in (0, 1, 2):
        np.add.at(vertex_normals, faces[:, col], normals)
    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    np.divide(vertex_normals, lengths, out=vertex_normals, where=lengths > 0)

    # Per-face AABBs
    bounds_min = np.minimum(np.minimum(p1, p2), p3)
    bounds_max = np.maximum(np.maximum(p1, p2), p3)

    m = len(faces)
    return Mesh(
        vertices=vertices,
        faces=faces,
        edge1=edge1,
        edge2=edge2,
        p1=p1,
        normals=normals,  # pyright: ignore[reportArgumentType]
        vertex_normals=vertex_normals,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        colors=np.broadcast_to(np.array(color, dtype=np.float32), (m, 3)).copy(),
        reflection=np.full(m, 0.5, dtype=np.float32),
    )
