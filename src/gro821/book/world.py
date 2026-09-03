from dataclasses import dataclass, field
from random import Random

from .geometry import Point


@dataclass
class World:
    """A rectangular 2D environment for computational geometry experiments."""

    width: float
    height: float
    rng: Random = field(default_factory=Random, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("World dimensions must be positive.")

    def generate_point(self) -> Point:
        """Generate a uniformly distributed point inside the world."""
        return Point(
            self.rng.uniform(0, self.width),
            self.rng.uniform(0, self.height),
        )

    def generate_points(self, count: int) -> list[Point]:
        """Generate ``count`` uniformly distributed points inside the world."""
        if not isinstance(count, int):
            raise TypeError("Point count must be an integer.")
        if count < 0:
            raise ValueError("Point count must not be negative.")
        return [self.generate_point() for _ in range(count)]

    def contains(self, point: Point) -> bool:
        """Return whether ``point`` lies inside the world's closed bounds."""
        return 0 <= point.x <= self.width and 0 <= point.y <= self.height
