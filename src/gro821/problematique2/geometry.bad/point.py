from dataclasses import dataclass
from math import nan


@dataclass(slots=True)
class Point:
    x: float = nan
    y: float = nan
    z: float = nan
