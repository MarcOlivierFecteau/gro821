#!/usr/bin/env python3

import math
import random
from typing import Final

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from numpy.typing import NDArray


class RobotConfig:
    def __init__(
        self,
        arm_width: float = 0.2,
        arm1_length: float = 0.75,
        arm2_length: float = 0.5,
        theta1: float = 0.0,
        theta2: float = 0.0,
    ) -> None:
        self.arm_width: Final = arm_width
        self.arm1_length: Final = arm1_length
        self.arm2_length: Final = arm2_length
        self.theta1 = theta1
        self.theta2 = theta2

    def __str__(self) -> str:
        return f"({self.arm_width}, {self.arm1_length}, {self.arm2_length}): ({self.theta1:1.02f}, {self.theta2:1.02f})"

    def __repr__(self) -> str:
        return f"RobotConfig({self.arm_width}, {self.arm1_length}, {self.arm2_length}, {self.theta1}, {self.theta2})"


def generate_rand_conf(l: float = 0.2, r2: float = 0.75, r3: float = 0.5) -> RobotConfig:
    theta1 = random.uniform(-np.pi, np.pi)
    theta2 = random.uniform(-np.pi, np.pi)
    return RobotConfig(l, r2, r3, theta1, theta2)


def generate_obstacles(
    clusters: int, points_per_obstacle: int, max_cluster_size: int, max_distance_from_center: int
):
    points: list[tuple[float, float]] = []
    for _ in range(clusters):
        cx = random.uniform(-max_distance_from_center, max_distance_from_center)
        cy = random.uniform(-max_distance_from_center, max_distance_from_center)
        cluster_radius = random.uniform(0, max_cluster_size)
        radius_error = 0.01 * cluster_radius
        for _ in range(points_per_obstacle):
            radius = cluster_radius + random.uniform(-radius_error, radius_error)
            theta = random.uniform(-np.pi, np.pi)
            px = cx + radius * math.cos(theta)
            py = cy + radius * math.sin(theta)
            points.append((px, py))
    return np.array(points, dtype=np.float64)


def rotation_matrix_2d(theta: float) -> NDArray[np.float64]:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )


def link_rectangle_points(
    origin: NDArray[np.float64], x_axis: NDArray[np.float64], height: float, width: float
) -> NDArray[np.float64]:
    rot = rotation_matrix_2d(-np.pi / 2)
    y_offset = height / 2 * rot @ x_axis
    x_offset = width * x_axis

    p1 = origin + y_offset
    p2 = p1 + x_offset
    p4 = origin - y_offset
    p3 = p4 + x_offset
    return np.array([p1, p2, p3, p4], dtype=np.float64)


def draw_obstacles(points: NDArray[np.float64]) -> None:
    _ = plt.scatter(points[:, 0], points[:, 1], c="blue")


def draw_robot_config(robot: RobotConfig) -> None:
    base = np.array([0, 0], dtype=np.float64).T
    rot_base_arm1 = rotation_matrix_2d(robot.theta1)
    rot_arm1_arm2 = rotation_matrix_2d(robot.theta2)
    j1_origin = base + rot_base_arm1 @ [0, 0]
    j1_x_axis = rot_base_arm1 @ [1, 0]
    j2_origin = j1_origin + robot.arm1_length * j1_x_axis
    j2_x_axis = rot_arm1_arm2 @ j1_x_axis

    tool_origin = j2_origin + robot.arm2_length * j2_x_axis

    arm1_link = link_rectangle_points(j1_origin, j1_x_axis, robot.arm_width, robot.arm1_length)
    arm2_link = link_rectangle_points(j2_origin, j2_x_axis, robot.arm_width, robot.arm2_length)

    # Draw second arm first to get correct depth rendering
    _ = plt.gca().add_patch(patches.Polygon(arm2_link, closed=True, color="green", fill=True))
    # Round the ends of each link so that the rectangular links are pill-shaped.
    link_radius = robot.arm_width / 2
    _ = plt.gca().add_patch(
        patches.Circle((tool_origin[0], tool_origin[1]), link_radius, color="green", fill=True)
    )

    # Draw end effector. NOTE: not considered in collision.
    end_effector = tool_origin + link_radius * j2_x_axis
    _ = plt.scatter(end_effector[0], end_effector[1], s=9.0, c="black")

    # Draw first arm last so it always overwrites the second arm and end effector
    _ = plt.gca().add_patch(
        patches.Polygon(arm1_link, closed=True, color="red", fill=True, zorder=3)
    )
    for center in (j1_origin, j2_origin):
        _ = plt.gca().add_patch(
            patches.Circle((center[0], center[1]), link_radius, color="red", fill=True, zorder=3)
        )
    _ = plt.gca().add_patch(
        patches.Circle((j2_origin[0], j2_origin[1]), link_radius, color="red", fill=True, zorder=3)
    )


def conf_is_valid(conf: RobotConfig, obstacles: NDArray[np.float64]) -> bool:
    base = np.array([0.0, 0.0], dtype=np.float64)
    arm1_axis = rotation_matrix_2d(conf.theta1) @ np.array([1.0, 0.0])
    arm2_axis = rotation_matrix_2d(conf.theta2) @ arm1_axis
    j2_origin = base + conf.arm1_length * arm1_axis

    links = (
        link_rectangle_points(base, arm1_axis, conf.arm_width, conf.arm1_length),
        link_rectangle_points(j2_origin, arm2_axis, conf.arm_width, conf.arm2_length),
    )

    # For a convex rectangle, a point is inside (or on its boundary) when
    # every edge cross product has the same sign.
    for corners in links:
        edges = np.roll(corners, -1, axis=0) - corners
        relative = obstacles[:, None, :] - corners[None, :, :]
        cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
        inside = np.all(cross >= 0.0, axis=1) | np.all(cross <= 0.0, axis=1)
        if np.any(inside):
            return False

    # The rounded ends of the links are circles with the same radius as the
    # link half-width.  Check the base, joint, and tool-end circles as well.
    circle_centers = np.array(
        [base, j2_origin, j2_origin + conf.arm2_length * arm2_axis], dtype=np.float64
    )
    radius_squared = (conf.arm_width / 2.0) ** 2
    offsets = obstacles[:, None, :] - circle_centers[None, :, :]
    return not np.any(np.sum(offsets * offsets, axis=2) <= radius_squared)


def main():
    # random.seed(42)  # To validate the logic with the same inputs between runs
    obstacles = generate_obstacles(10, 20, 1, 3)
    for i in range(3):
        plt.figure(i)
        draw_obstacles(obstacles)
        robot_conf = generate_rand_conf()
        draw_robot_config(robot_conf)
        plt.show(block=False)
        print(f"{robot_conf!s} valid? {conf_is_valid(robot_conf, obstacles)}")
    plt.show(block=True)


if __name__ == "__main__":
    main()
