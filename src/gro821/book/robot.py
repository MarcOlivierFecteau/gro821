import random
from dataclasses import dataclass
from typing import Final

import numpy as np

from .geometry import Circle, Point


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
    base: Point | tuple[float, float] = (0, 0), l: float = 0.2, r2: float = 0.75, r3: float = 0.5
) -> RobotConfig:
    theta1 = random.uniform(-np.pi, np.pi)
    theta2 = random.uniform(-np.pi, np.pi)
    if isinstance(base, Point):
        return RobotConfig(base, l, r2, r3, theta1, theta2)
    return RobotConfig(Point(*base), l, r2, r3, theta1, theta2)
