#!/usr/bin/env python3

import matplotlib.pyplot as plt

from .algorithms import convex_hull as CH
from .render import matplotlib_renderer as renderer
from .robot import generate_rand_conf, get_navigable_area
from .world import World


def main() -> None:
    world = World(5, 5)
    points = world.generate_points(10)
    convex_hull = CH.slow_convex_hull(points, "ccw")
    # robot_conf = generate_rand_conf((world.width / 2, world.height / 2))
    # navigable_area = get_navigable_area(robot_conf)

    _ = renderer.draw_world(world)
    _ = renderer.draw_points(points)
    _ = renderer.draw_convex_hull(convex_hull)
    # renderer.draw_robot_config(robot_conf)
    # renderer.draw_navigable_area(navigable_area)

    plt.show(block=True)


if __name__ == "__main__":
    main()
