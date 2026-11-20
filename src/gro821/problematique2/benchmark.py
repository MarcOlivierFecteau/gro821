from pathlib import Path

import numpy as np

from gro821.problematique2.main import main as _main

SHADING = ["flat", "gouraud"]
ACCELERATION = ["bvh", "bsp", "octree", "root_aabb"]
LIGHTPOS = [
    np.array([5, 5, 10], dtype=np.float32),
    np.array([5, 5, 30], dtype=np.float32),
    np.array([5, 5, -10], dtype=np.float32),
]
CULLING = [True, False]
EARLY_EXIT = [True, False]
VECTORIZED = [True, False]


def main():
    root = Path(__file__).parent
    model = f"{root}/meshes/bun_zipper.ply"
    for shade in SHADING:
        for accel in ACCELERATION:
            for cul in CULLING:
                name_col = "c1" if cul else "c0"
                for early in EARLY_EXIT:
                    name_early = "e1" if early else "e0"
                    for vector in VECTORIZED:
                        name_vec = "v1" if vector else "v0"
                        name = (
                            shade
                            + "_"
                            + accel
                            + "_"
                            + str(10)
                            + "_"
                            + name_col
                            + "_"
                            + name_early
                            + "_"
                            + name_vec
                        )
                        _main(model, shade, accel, LIGHTPOS[0], cul, early, vector, name)


if __name__ == "__main__":
    main()
