from dataclasses import dataclass
from io import SEEK_CUR, SEEK_END, SEEK_SET, TextIOWrapper, _WrappedBuffer
from os import path
from typing import Final

from gro821.problematique2.geometry.face import Face
from gro821.problematique2.ply.mesh import Mesh

PLY_SCALAR_TYPES: Final[dict[str, tuple[str, type]]] = {
    # PLY type: (binary format, Python type)
    "char": ("b", int),
    "uchar": ("B", int),
    "short": ("h", int),
    "ushort": ("H", int),
    "int": ("i", int),
    "uint": ("I", int),
    "float": ("f", float),
    "double": ("d", float),
}


@dataclass(slots=True)
class Property:
    name: str
    value_type: type
    item_type: type | None
    count_type: type | None


@dataclass(slots=True)
class Header:
    format: str | None = None
    vertex_count: int = -1
    vertex_properties: list[Property] | None = None
    face_count: int = -1
    face_properties: list[Property] | None = None


def parse_header(stream: TextIOWrapper[_WrappedBuffer]) -> Header:
    filename = path.abspath(stream.name)
    assert isinstance(filename, str)

    current_element: str = ""

    header = Header()
    for i, line in enumerate(stream):
        if i == 0 and line != "ply":
            raise OSError("The first line of the file is not 'ply'")
        if line.startswith("comment"):
            continue
        if line.startswith("format"):
            if header.format is None:
                header.format = " ".join(line.split(" ")[1:]).lower()
                ply_format_line = i
                if header.format != "ascii 1.0":
                    raise TypeError(f"Unsupported PLY variation: {header.format}")
            else:
                # `header_format_line` is necessarily bound
                raise OSError(
                    f"({filename}:{i}) Format is already specified in line {ply_format_line}"  # pyright: ignore[reportPossiblyUnboundVariable]
                )
        if line.startswith("element"):
            words = line.split(" ")[1:]  # Consume token 'element'
            current_element = words[0].lower()
            if current_element == "vertex":
                if header.vertex_count is None:
                    header.vertex_count = int(words[1])
                    vertex_count_line = i
                else:
                    raise OSError(
                        f"({filename}:{i}) Vertex count is already specified in line {vertex_count_line}"  # pyright: ignore[reportPossiblyUnboundVariable]
                    )
            elif current_element == "face":
                if header.face_count is None:
                    header.face_count = int(words[1])
                    face_count_line = i
                else:
                    raise OSError(
                        f"({filename}:{i}) Face count is already specified in line {face_count_line}"  # pyright: ignore[reportPossiblyUnboundVariable]
                    )
            else:
                raise OSError(f"Unsupported element type: {current_element}")
        if line.startswith("property"):
            # TODO: check for a way to count the number of properties to statically initialize `properties` field in `header`
            words = line.split(" ")[1:]  # Consume token 'property'
            property_type = words[0].lower()
            if property_type == "list":
                # Most likely processing 'element face'
                if current_element != "face":
                    raise OSError(
                        "'list' properties are only supported for element 'face'"
                    )
                if header.face_properties is None:
                    header.face_properties = []

                # NOTE: face properties's schema is assumed
                assert len(words) == 4, "Too many properties"
                count_type = words[1].lower()
                if count_type not in PLY_SCALAR_TYPES:
                    raise OSError(f"Unsupported property type: {count_type}")
                item_type = words[2].lower()
                if item_type not in PLY_SCALAR_TYPES:
                    raise OSError(f"Unsupported property type: {item_type}")

                face_property = Property(
                    name=words[-1].lower(),
                    value_type=list,
                    item_type=PLY_SCALAR_TYPES[item_type][1],
                    count_type=PLY_SCALAR_TYPES[count_type][1],
                )
                header.face_properties.append(face_property)

            elif property_type not in PLY_SCALAR_TYPES:
                raise OSError(f"Unsupported property type: {property_type}")

            if current_element == "vertex":
                if header.vertex_properties is None:
                    header.vertex_properties = []
                vertex_property = Property(
                    name=words[2].lower(),
                    value_type=PLY_SCALAR_TYPES[property_type][1],
                    item_type=None,
                    count_type=None,
                )
                header.vertex_properties.append(vertex_property)
        if line.startswith("end_header"):
            assert header.format is not None
            assert header.vertex_count >= 0
            assert (
                header.vertex_properties is not None
                and len(header.vertex_properties) >= 0
            )
            assert header.face_count >= 0
            assert (
                header.face_properties is not None and len(header.face_properties) >= 0
            )
            return header
    raise OSError(f"'end_header' not found in {filename}")


def parse_records(
    stream: TextIOWrapper[_WrappedBuffer], header: Header, scale: float = 1.0
) -> Mesh:
    """NOTE: Assumed there are no blank lines."""
    assert stream.seek(0, SEEK_SET) == 0
    while not stream.readline().startswith("end_header"):
        continue

    # === Parse vertex records ===

    point_variable_names = {"x", "y", "z"}
    point_indices: dict[str, int] = {}
    assert header.vertex_properties is not None
    for i, prop in enumerate(header.vertex_properties):
        if prop.name in point_variable_names and prop.value_type == float:
            point_variable_names.remove(prop.name)
            point_indices[prop.name] = i
    point_requirements_met = len(point_variable_names) == 0
    auxiliary_properties = len(header.vertex_properties) > len(point_variable_names)
    del point_variable_names

    vertices_stream = stream.readlines(header.vertex_count)
    assert len(vertices_stream) == header.vertex_count

    aux_vertex_indices: list[int] = []
    if auxiliary_properties:
        record = stream.readline()
        assert stream.seek(0, SEEK_SET) == 0
        aux_vertex_indices = sorted(
            {i for i in range(len(record))}.difference(point_indices)
        )
        mesh = Mesh(
            header.vertex_count,
            header.face_count,
            {
                header.vertex_properties[i].name: [
                    header.vertex_properties[i].value_type()
                ]
                * header.vertex_count
                for i in aux_vertex_indices
            },
        )
    else:
        mesh = Mesh(header.vertex_count, header.face_count, {})
    assert mesh.vertex_data is not None

    if point_requirements_met:
        for v, line in enumerate(vertices_stream):
            record = line.split(" ")
            x = record[point_indices["x"]]
            y = record[point_indices["y"]]
            z = record[point_indices["z"]]
            mesh.vertices[v].x = scale * float(x)
            mesh.vertices[v].y = scale * float(y)
            mesh.vertices[v].z = scale * float(z)
        if auxiliary_properties:
            for v, line in enumerate(vertices_stream):
                record = line.split(" ")
                for i in aux_vertex_indices:
                    prop = header.vertex_properties[i]
                    mesh.vertex_data[prop.name][v] = prop.value_type(record[i])
    else:
        raise OSError("Unsupported combination of vertex properties")

    # === Parse face records ===

    assert header.face_properties is not None
    faces_stream = stream.readlines(header.face_count)
    assert len(faces_stream) == header.face_count

    for f, line in enumerate(faces_stream):
        # TODO: make more robust
        record = line.split(" ")
        prop = header.face_properties[0]
        vertex_count = prop.count_type(record[0])  # pyright: ignore[reportOptionalCall]
        assert vertex_count == len(record[1:]) and vertex_count == len(
            Face.vertex_indices
        )

        face = Face(
            tuple(prop.item_type(record[i + 1]) for i in range(vertex_count)),  # pyright: ignore[reportOptionalCall]
        )
        if any(i < 0 or i >= header.vertex_count for i in face.vertex_indices):
            raise ValueError(f"Vertex index out of range: (face[{f}])")
        mesh.faces[f] = face

    while stream.readline() != "":
        continue

    assert stream.seek(0, SEEK_CUR) == stream.seek(0, SEEK_END)
    return mesh


def load_ply(filename: str, scale: float = 1.0) -> Mesh:
    with open(filename, "r") as file:
        header = parse_header(file)
        mesh = parse_records(file, header, scale)
        return mesh
