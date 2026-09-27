import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes

from ..geometry import (
    AABB,
    OBB,
    Circle,
    Point,
)
from ..robot import RobotConfig
from ..structures.kdtree import KdTreeNode
from ..structures.quadtree import CompressedQuadtreeNode, PointQuadtreeNode, SpaceQuadtreeNode
from ..world import World


def draw_world(world: World) -> Axes:
    """Draw ``world`` and return the axes used for the rendering."""
    figure = plt.figure("World")
    figure.clear()
    axes = figure.add_subplot(111)
    axes.set_xlabel(r"$\hat{x}_w$")
    axes.set_ylabel(r"$\hat{y}_w$")
    axes.set_xlim(-0.1 * world.width, 1.1 * world.width)
    axes.set_ylim(-0.1 * world.height, 1.1 * world.height)
    axes.set_aspect("equal")
    axes.hlines([0, world.height], [0, 0], [world.width, world.width], colors="black")
    axes.vlines([0, world.width], [0, 0], [world.height, world.height], colors="black")
    return axes


def draw_points(points: list[Point], zorder: int = 1) -> Axes:
    """Draw points on the current axes in red."""
    axes = plt.gca()
    axes.scatter(
        [point.x for point in points], [point.y for point in points], s=5.0, c="red", zorder=zorder
    )
    return axes


def draw_robot_config(robot: RobotConfig) -> Axes:
    """Draw a robot configuration on the current axes."""
    axes = plt.gca()
    arm1_axis = np.array([np.cos(robot.theta1), np.sin(robot.theta1)], dtype=np.float64)
    arm2_angle = robot.theta1 + robot.theta2
    arm2_axis = np.array([np.cos(arm2_angle), np.sin(arm2_angle)], dtype=np.float64)
    base = np.array([robot.base.x, robot.base.y], dtype=np.float64)
    joint = base + robot.arm1_length * arm1_axis
    end = joint + robot.arm2_length * arm2_axis
    radius = robot.arm_width / 2
    end_effector = end + radius * arm2_axis

    axes.add_patch(
        patches.Rectangle(
            tuple(joint - radius * np.array([-arm2_axis[1], arm2_axis[0]])),
            robot.arm2_length,
            robot.arm_width,
            angle=np.degrees(arm2_angle),
            color="green",
        )
    )
    axes.add_patch(patches.Circle(tuple(end), radius, color="green"))
    axes.scatter(end_effector[0], end_effector[1], s=9.0, c="black")

    axes.add_patch(
        patches.Rectangle(
            tuple(base - radius * np.array([-arm1_axis[1], arm1_axis[0]])),
            robot.arm1_length,
            robot.arm_width,
            angle=np.degrees(robot.theta1),
            color="red",
            zorder=3,
        )
    )
    for center in (base, joint):
        axes.add_patch(patches.Circle(tuple(center), radius, color="red", zorder=3))
    return axes


def draw_navigable_area(navigable_area: tuple[Circle, Circle]) -> Axes:
    """Draw the inner and outer circles of a robot's navigable area."""
    axes = plt.gca()
    inner, outer = navigable_area
    axes.add_patch(
        patches.Circle(
            (outer.base.x, outer.base.y),
            outer.radius,
            edgecolor="green",
            facecolor=(0.0, 0.5, 0.0, 0.2),
            fill=True,
        )
    )
    axes.add_patch(
        patches.Circle(
            (inner.base.x, inner.base.y),
            inner.radius,
            edgecolor="green",
            facecolor="white",
            fill=True,
        )
    )
    return axes


def draw_convex_hull(convex_hull: list[Point]) -> Axes:
    """Draw the convex hull as a closed polygon."""
    axes = plt.gca()
    vertices = [(point.x, point.y) for point in convex_hull]
    axes.add_patch(
        patches.Polygon(
            vertices,
            closed=True,
            fill=False,
            edgecolor="blue",
            linewidth=1.5,
        )
    )
    return axes


def draw_quadtree(quadtree: SpaceQuadtreeNode | PointQuadtreeNode) -> Axes:
    """Draw the leaf regions of ``quadtree`` on the current axes."""
    axes = plt.gca()

    if quadtree.is_leaf():
        start, end = quadtree.bb.start, quadtree.bb.end
        axes.add_patch(
            patches.Rectangle(
                (start.x, start.y),
                end.x - start.x,
                end.y - start.y,
                fill=False,
                edgecolor="gray",
                linewidth=0.5,
            )
        )
        return axes

    for child in (quadtree.NW, quadtree.NE, quadtree.SW, quadtree.SE):
        if child is not None:
            draw_quadtree(child)
    return axes


def draw_kdtree(kdtree: KdTreeNode) -> Axes:
    """Draw the leaf regions of ``kdtree`` on the current axes."""
    axes = plt.gca()

    if kdtree.is_leaf():
        start, end = kdtree.bb.start, kdtree.bb.end
        axes.add_patch(
            patches.Rectangle(
                (start.x, start.y),
                end.x - start.x,
                end.y - start.y,
                fill=False,
                edgecolor="gray",
                linewidth=0.5,
            )
        )
        return axes

    if kdtree.lhs is not None:
        draw_kdtree(kdtree.lhs)
    if kdtree.rhs is not None:
        draw_kdtree(kdtree.rhs)
    return axes


def draw_compressed_quadtree(quadtree: CompressedQuadtreeNode) -> Axes:
    """Draw the represented regions of a compressed quadtree on the current axes."""
    axes = plt.gca()

    start, end = quadtree.bb.start, quadtree.bb.end
    axes.add_patch(
        patches.Rectangle(
            (start.x, start.y),
            end.x - start.x,
            end.y - start.y,
            fill=False,
            edgecolor="gray",
            linewidth=0.5,
        )
    )

    if not quadtree.is_leaf():
        for child in quadtree.children.values():
            draw_compressed_quadtree(child)
    return axes


def draw_aabb(bb: AABB, color: tuple[float, float, float]) -> Axes:
    """Draw the axis-aligned bounding box as a rectangle."""
    axes = plt.gca()
    start, end = bb.start, bb.end
    axes.add_patch(
        patches.Rectangle(
            (start.x, start.y),
            end.x - start.x,
            end.y - start.y,
            edgecolor=color,
            facecolor=(*color, 0.2),
            fill=True,
            linewidth=0.5,
            zorder=10,
        )
    )
    return axes


def draw_obb(bb: OBB, color: tuple[float, float, float]) -> Axes:
    """Draw the oriented bounding box as a rotated polygon."""
    axes = plt.gca()
    vertices = [(point.x, point.y) for point in bb.corners()]
    axes.add_patch(
        patches.Polygon(
            vertices,
            closed=True,
            edgecolor=color,
            facecolor=(*color, 0.2),
            fill=True,
            linewidth=0.5,
            zorder=10,
        )
    )
    return axes
