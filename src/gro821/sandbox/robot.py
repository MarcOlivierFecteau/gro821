from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from gro821.sandbox.geometry import (
    AABB,
    OBB,
    Circle,
    Edge,
    Point,
    Vec2,
    point_to_edge_distance_squared,
)

if TYPE_CHECKING:
    from gro821.sandbox.world import World


@dataclass
class RobotConfig:
    base: Final[Point]
    arm_width: Final[float]
    arm1_length: Final[float]
    arm2_length: Final[float]
    theta1: float
    theta2: float

    def __post_init__(self) -> None:
        if any(
            [self.theta1 < -np.pi, self.theta1 > np.pi, self.theta2 < -np.pi, self.theta2 > np.pi]
        ):
            raise ValueError("Joint angles must be between -π and π.")

    def __str__(self) -> str:
        return f"({self.arm_width}, {self.arm1_length}, {self.arm2_length}): ({self.theta1:1.02f}, {self.theta2:1.02f})"

    def __repr__(self) -> str:
        return f"RobotConfig({self.arm_width}, {self.arm1_length}, {self.arm2_length}, {self.theta1}, {self.theta2})"


def get_navigable_area(robot: RobotConfig) -> tuple[Circle, Circle]:
    """Return the inner and outer circles delimiting the end-effector area."""
    return (
        Circle(robot.base, abs(robot.arm1_length - robot.arm2_length - robot.arm_width / 2)),
        Circle(robot.base, robot.arm1_length + robot.arm2_length + robot.arm_width / 2),
    )


def generate_rand_conf(
    base: Point, l: float = 0.2, r2: float = 0.75, r3: float = 0.5
) -> RobotConfig:
    theta1 = random.uniform(-np.pi, np.pi)
    theta2 = random.uniform(-np.pi, np.pi)
    if isinstance(base, Point):
        return RobotConfig(base, l, r2, r3, theta1, theta2)
    return RobotConfig(Point(*base), l, r2, r3, theta1, theta2)


def generate_rand_conf_prop_world(world: World, base: Point | None = None) -> RobotConfig:
    """Generates a random robot configuration with dimensions proportional to the world's dimensions."""
    if base is None:
        base = Point(world.width / 2, world.height / 2)
    l = min(world.width, world.height) / 15
    r2 = min(world.width, world.height) / 4
    r3 = min(world.width, world.height) / 6
    return generate_rand_conf(base, l, r2, r3)


def get_link_aabb(robot: RobotConfig, link: int) -> AABB:
    if link not in {0, 1}:
        raise ValueError("`link` must be either 0 (arm1) or 1 (arm2).")
    half_width = robot.arm_width / 2
    if link == 0:
        link_axis = Point(math.cos(robot.theta1), math.sin(robot.theta1))
        center_start = robot.base
        center_end = center_start + robot.arm1_length * link_axis
    else:
        arm1_axis = Point(math.cos(robot.theta1), math.sin(robot.theta1))
        link_axis = Point(
            math.cos(robot.theta1 + robot.theta2), math.sin(robot.theta1 + robot.theta2)
        )
        center_start = robot.base + robot.arm1_length * arm1_axis
        center_end = center_start + robot.arm2_length * link_axis

    normal = Point(-link_axis.y, link_axis.x)
    corners = [
        center_start + half_width * normal,
        center_start - half_width * normal,
        center_end + half_width * normal,
        center_end - half_width * normal,
    ]
    cap_centers = [center_start, center_end]
    for center in cap_centers:
        corners.extend(
            [
                Point(center.x - half_width, center.y - half_width),
                Point(center.x + half_width, center.y + half_width),
            ]
        )
    return AABB(
        Point(min(point.x for point in corners), min(point.y for point in corners)),
        Point(max(point.x for point in corners), max(point.y for point in corners)),
    )


def get_link_obb(robot: RobotConfig, link: int) -> OBB:
    """Return the tight oriented box enclosing a robot link's pill shape."""
    if link not in {0, 1}:
        raise ValueError("`link` must be either 0 (arm1) or 1 (arm2).")
    if link == 0:
        axis = Point(math.cos(robot.theta1), math.sin(robot.theta1))
        start = robot.base
        length = robot.arm1_length
    else:
        arm1_axis = Point(math.cos(robot.theta1), math.sin(robot.theta1))
        axis = Point(math.cos(robot.theta1 + robot.theta2), math.sin(robot.theta1 + robot.theta2))
        start = robot.base + robot.arm1_length * arm1_axis
        length = robot.arm2_length
    end = start + length * axis
    return OBB(
        (start + end) * 0.5,
        Point((length + robot.arm_width) / 2, robot.arm_width / 2),
        math.atan2(axis.y, axis.x),
    )


def get_link_edges(robot: RobotConfig) -> tuple[Edge, Edge]:
    l1_axis = Vec2(math.cos(robot.theta1), math.sin(robot.theta1))
    l2_axis = Vec2(math.cos(robot.theta1 + robot.theta2), math.sin(robot.theta1 + robot.theta2))
    j1 = robot.base + robot.arm1_length * l1_axis
    l1: Edge = (robot.base, j1)
    l2: Edge = (j1, j1 + robot.arm2_length * l2_axis)
    return (l1, l2)


def point_collides_edge(p: Point, link: Edge, radius: float, epsilon: float = 1e-6) -> bool:
    return point_to_edge_distance_squared(p, link, epsilon) <= radius * radius + epsilon


def conf_is_valid_naive(robot: RobotConfig, points: list[Point]) -> bool:
    links = get_link_edges(robot)
    for p in points:
        if any(point_collides_edge(p, l, robot.arm_width) for l in links):
            return False
    return True
