from collections import deque
from collections.abc import Mapping
from sys import getsizeof
from typing import Any

_CONTAINER_TYPES = (list, tuple, set, frozenset, deque)


def retained_size(root: object) -> int:
    """Return the retained size of the reachable object graph rooted at ``root``.

    The result is the sum of ``sys.getsizeof`` for each reachable object. Objects
    referenced more than once are counted once, so shared points and bounds do not
    inflate the result. This measures Python object memory for the active runtime,
    not a portable serialized or native data-structure size.
    """
    total, _ = _walk(root, set(), None)
    return total


def retained_size_breakdown(root: object) -> dict[str, int]:
    """Return retained bytes grouped by the concrete type name of each object."""
    _, breakdown = _walk(root, set(), {})
    return breakdown


def _walk(
    value: object,
    seen: set[int],
    breakdown: dict[str, int] | None,
) -> tuple[int, dict[str, int]]:
    object_id = id(value)
    if object_id in seen:
        return 0, breakdown or {}
    seen.add(object_id)

    size = getsizeof(value)
    if breakdown is not None:
        type_name = type(value).__name__
        breakdown[type_name] = breakdown.get(type_name, 0) + size

    total = size
    for referent in _referents(value):
        referent_size, breakdown = _walk(referent, seen, breakdown)
        total += referent_size
    return total, breakdown or {}


def _referents(value: object) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        referents: list[Any] = []
        for key, item in value.items():
            referents.extend((key, item))
        return tuple(referents)

    if isinstance(value, _CONTAINER_TYPES):
        return tuple(value)

    attributes = getattr(value, "__dict__", None)
    if attributes is not None:
        return (attributes,)

    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    return tuple(
        getattr(value, slot)
        for slot in slots
        if slot not in {"__dict__", "__weakref__"} and hasattr(value, slot)
    )
