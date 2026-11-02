from typing import Any

from gro821.problematique2.geometry.face import Face
from gro821.problematique2.geometry.point import Point


class Mesh:
    __slots__ = ["faces", "vertex_data", "vertices"]

    def __init__(
        self,
        vertices: list[Point] | int,
        faces: list[Face] | int,
        vertex_data: dict[str, list[Any]] | None = None,
    ):
        if isinstance(vertices, int):
            self.vertices = [Point()] * vertices
        elif isinstance(vertices, list):
            if len(vertices) > 0:
                if isinstance(vertices[0], Point):
                    self.vertices = vertices
                else:
                    raise TypeError(
                        f"Unsupported type for `vertices[0]`: {type(vertices[0])}"
                    )
            else:
                raise ValueError("Empty list not supported for `vertices`")
        else:
            raise TypeError(f"Unsupported type for `vertices`: {type(vertices)}")
        if isinstance(faces, int):
            self.faces = [Face()] * faces
        elif isinstance(faces, list):
            if len(faces) > 0:
                if isinstance(faces[0], Face):
                    self.faces = faces
                else:
                    raise TypeError(
                        f"Unsupported type for `faces[0]`: {type(faces[0])}"
                    )
            else:
                raise ValueError("Empty list not supported for `faces`")
        else:
            raise TypeError(f"Unsupported type for `faces`: {type(faces)}")
        self.vertex_data = vertex_data
