"""
Re-implementation of `bunny.py` from the provided codebase.
"""

from enum import Enum, auto
from pathlib import Path

import numpy as np

root = Path(__file__).parent


class State(Enum):
    GET_VC = auto()  # Get Vertex Count
    GET_FC = auto()  # Get Face Count
    FND_HD = auto()  # FiND end of HeaDer
    GET_VX = auto()  # Get Vertices
    GET_FS = auto()  # Get Faces
    DONE = auto()


def load(filename: str, scale: float = 1.0):
    with open(filename, "r") as file:
        vertex_count: int = 0
        face_count: int = 0
        _vertices: list[tuple[float, float, float]] = []
        _faces: list[tuple[int, int, int]] = []

        state = State.GET_VC
        for line in file:
            match state:
                case State.GET_VC:
                    if line.startswith("element vertex"):
                        _, _, vcs = line.split(" ")
                        vertex_count = int(vcs)
                        state = State.GET_FC
                case State.GET_FC:
                    if line.startswith("element face"):
                        _, _, fcs = line.split(" ")
                        face_count = int(fcs)
                        state = State.FND_HD
                case State.FND_HD:
                    if line.startswith("end_header"):
                        state = State.GET_VX
                case State.GET_VX:
                    x, y, z = line.split(" ")[:3]
                    _vertices.append((scale * float(x), scale * float(y), scale * float(z)))
                    vertex_count -= 1
                    if vertex_count <= 0:
                        state = State.GET_FS
                case State.GET_FS:
                    i1, i2, i3 = line.split(" ")[1:4]
                    _faces.append((int(i1), int(i2), int(i3)))
                    face_count -= 1
                    if face_count <= 0:
                        state = State.DONE
        if state != State.DONE:
            raise RuntimeError(f"Stuck in {state.name}.")

        # == Better data model: flat NumPy arrays (memory & compute)
        # A (N, 3) f32 array is exactly 4N x 3 contiguous bytes: no per-object
        # headers, no pointer indirection. A model with a high number of vertices
        # drops from MB of Python objects to a few hundred KB of raw data. We use
        # f32 because 7 significant digits suffice for coordinates in a scene that
        # spans ~10 units.
        # Contiguous, homogeneous data is what NumPy, the CPU cache, and SIMD are
        # optimized for. And crucially, the conversions in `trace_ray` disappear:
        # `vertices[face[i]]` is a fancy-indexing operation on a C array; one step,
        # no `np.array(...)` construction, no boxing. Moreover, Python function-
        # call overhead dominates small-array operations, which is the case here.
        vertices = np.array(_vertices, dtype=np.float32)  # Shape (N, 3)
        faces = np.array(_faces, dtype=np.int32)  # Shape (M, 3)

        return vertices, faces


if __name__ == "__main__":
    for file in (
        "bun_zipper.ply",
        "bun_zipper_res2.ply",
        "bun_zipper_res3.ply",
        "bun_zipper_res4.ply",
    ):
        vertices, faces = load(f"{root}/meshes/{file}")
        print(f"{file}: got {len(vertices)} vertices and {len(faces)} faces.\n")
