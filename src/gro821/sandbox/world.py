import math
from dataclasses import dataclass, field
from random import Random

from gro821.sandbox.geometry import Point
from gro821.sandbox.robot import RobotConfig, get_link_aabb


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

    def generate_obstacles(
        self,
        clusters: int,
        points_per_cluster: int,
        max_cluster_radius: float,
        max_dist_from_center: float,
        robot: RobotConfig | None = None,
    ) -> list[Point]:
        """
        Generate uniformly distributed circular clusters of points from the center of the world.

        Args:
            robot: if given, ensures all generated points are outside the configuration.
        """
        if not isinstance(clusters, int) or clusters < 1:
            raise TypeError("Cluster count must be a positive integer.")
        if not isinstance(points_per_cluster, int) or points_per_cluster < 1:
            raise TypeError("Number of points per cluster must be a positive integer.")

        result: list[Point] = []
        world_center = Point(self.width / 2, self.height / 2)

        def gen_cluster_center() -> Point:
            return world_center + Point(
                self.rng.uniform(-max_dist_from_center, max_dist_from_center),
                self.rng.uniform(-max_dist_from_center, max_dist_from_center),
            )

        def gen_cluster_point() -> Point:
            radius = cluster_radius + radius_error * self.rng.uniform(-1, 1)
            theta = self.rng.uniform(-math.pi, math.pi)
            return cluster_center + radius * Point(math.cos(theta), math.sin(theta))

        if robot is not None:
            arm1_bb = get_link_aabb(robot, 0)
            arm2_bb = get_link_aabb(robot, 1)

        for _ in range(clusters):
            if robot is None:
                cluster_center = gen_cluster_center()
            else:  # Ensure the cluster center is outside initial configuration
                cluster_center = gen_cluster_center()
                while any([arm1_bb.contains(cluster_center), arm2_bb.contains(cluster_center)]):  # pyright: ignore[reportPossiblyUnboundVariable]
                    cluster_center = gen_cluster_center()

            cluster_radius = self.rng.uniform(0, max_cluster_radius)
            radius_error = 0.01 * cluster_radius
            valid_points: list[Point] = []

            for _ in range(points_per_cluster):
                if robot is None:
                    p = gen_cluster_point()
                else:
                    p = gen_cluster_point()
                    while any([arm1_bb.contains(p), arm2_bb.contains(p)]):  # pyright: ignore[reportPossiblyUnboundVariable]
                        p = gen_cluster_point()

                if 0 <= p.x <= self.width and 0 <= p.y <= self.height:
                    valid_points.append(p)

            result.extend(valid_points)
        return result
