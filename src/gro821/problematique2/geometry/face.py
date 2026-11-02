from dataclasses import dataclass
from typing import Final

from gro821.problematique2.geometry import Vec3

__NULL_NORMAL: Final = Vec3(0, 0, 0)


@dataclass(slots=True)
class Face:
    vertex_indices: tuple[int, int, int] = (0, 0, 0)
    normal: Vec3 | None = __NULL_NORMAL
