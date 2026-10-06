#!/usr/bin/env python3

from collections.abc import Callable
from time import perf_counter_ns

import matplotlib.pyplot as plt

import gro821.sandbox.structures.quadtree as qd
from gro821.sandbox import fmt
from gro821.sandbox.algorithms import convex_hull as CH
from gro821.sandbox.benchmark import (
    benchmark_methods,
    collision_methods,
    generate_collision_workloads,
)
from gro821.sandbox.comparison import ComparisonConfig, ComparisonResults, TimeResults
from gro821.sandbox.fmt import ANSI
from gro821.sandbox.geometry import Point, make_aabb_from_circle
from gro821.sandbox.render import matplotlib_renderer as renderer
from gro821.sandbox.render.plots import plot_time_results
from gro821.sandbox.robot import (
    RobotConfig,
    conf_is_valid_naive,
    generate_rand_conf_prop_world,
    get_link_aabb,
    get_link_obb,
    get_navigable_area,
)
from gro821.sandbox.structures import kdtree as kd
from gro821.sandbox.world import World


def run_collision_experiment(
    world: World,
    sample_count: int,
    clusters: int,
    points_per_cluster: int,
    *,
    bb_type: str = "aabb",
    capacity: int = 1,
) -> ComparisonResults:
    """Run all collision methods over one shared set of generated workloads."""
    workloads = generate_collision_workloads(
        world,
        sample_count,
        clusters,
        points_per_cluster,
    )
    return benchmark_methods(
        collision_methods(bb_type=bb_type, capacity=capacity),
        workloads,
    )


def compare_convex_hull(
    world: World, sample_size: int, num_points: int
) -> tuple[TimeResults, list[Point]]:
    results: list[TimeResults] = []
    convex_hull: list[Point] = []
    for _ in range(sample_size):
        points = world.generate_points(num_points)
        slow_start = perf_counter_ns()
        _ = CH.slow_convex_hull(points, "ccw")
        slow_end = perf_counter_ns()
        start = perf_counter_ns()
        convex_hull = CH.convex_hull(points)
        end = perf_counter_ns()
        rel_time = (end - start) / (slow_end - slow_start)
        speed_factor = 1 / rel_time
        results.append(TimeResults(slow_end - slow_start, end - start, speed_factor))
    avg_slow = avg_time = 0
    avg_factor = 0.0
    for slow_time, time, factor in results:
        avg_slow += slow_time
        avg_time += time
        avg_factor += factor
    avg_slow /= sample_size
    avg_time /= sample_size
    avg_factor /= sample_size
    print(f"Average Performance of Convex Hull\n({sample_size} samples of {num_points} points)")
    print(f"SlowConvexHull: {avg_slow} ns")
    print(f"ConvexHull: {avg_time} ns ({(avg_factor):.3f}x faster)")
    return (TimeResults(avg_slow, avg_time, avg_factor), convex_hull)


def compare_quadtree_to_naive(
    world: World,
    sample_size: int,
    num_points: int,
    bb_type: str,
    split_method: str,
    compressed: bool = False,
    robot: RobotConfig | None = None,
) -> tuple[TimeResults, list[Point]]:
    if bb_type.lower() not in {"aabb", "obb"}:
        raise ValueError("`bb_type` must be either 'aabb' or 'obb'.")
    if split_method.lower() not in {"point", "space"}:
        raise ValueError("`split_method` must be either 'point' or 'space'.")
    if split_method.lower() == "point" and compressed:
        raise ValueError("Invalid `split_method` for `compressed`.")

    robot_override = robot is not None

    results: list[TimeResults] = []
    if any(
        [
            sample_size <= 0,
            num_points <= 0,
            not isinstance(sample_size, int),
            not isinstance(num_points, int),
        ]
    ):
        raise ValueError("`sample_size` and `num_points` take positive integer values.")

    for _ in range(sample_size):
        # Keep robot roughly proportional to the world
        if not robot_override:
            robot = generate_rand_conf_prop_world(world)
        points = world.generate_points(num_points)
        robot_envelope = make_aabb_from_circle(get_navigable_area(robot)[1])  # pyright: ignore[reportArgumentType]
        if split_method.lower() == "point":
            quadtree = qd.build_point_quadtree(
                points, robot_envelope, 1
            )  # Worst case: always split
        else:
            quadtree = qd.build_space_quadtree(
                points, robot_envelope, 1
            )  # Worst case: always split
            if compressed:
                quadtree = qd.compress_quadtree(quadtree)

        if bb_type.lower() == "aabb":
            arm1_bb = get_link_aabb(robot, 0)  # pyright: ignore[reportArgumentType]
            arm2_bb = get_link_aabb(robot, 1)  # pyright: ignore[reportArgumentType]
        else:
            arm1_bb = get_link_obb(robot, 0)  # pyright: ignore[reportArgumentType]
            arm2_bb = get_link_obb(robot, 1)  # pyright: ignore[reportArgumentType]

        naive_start = perf_counter_ns()
        _ = conf_is_valid_naive(robot, points)  # pyright: ignore[reportArgumentType]
        naive_end = perf_counter_ns()

        if bb_type.lower() == "aabb":
            if split_method.lower() == "point":
                quadtree_start = perf_counter_ns()
                _ = qd.point_quadtree_collides_aabb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                _ = qd.point_quadtree_collides_aabb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                quadtree_end = perf_counter_ns()
            else:  # Space
                if compressed:
                    quadtree_start = perf_counter_ns()
                    _ = qd.compressed_quadtree_collides_aabb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                    _ = qd.compressed_quadtree_collides_aabb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                    quadtree_end = perf_counter_ns()
                else:
                    quadtree_start = perf_counter_ns()
                    _ = qd.space_quadtree_collides_aabb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                    _ = qd.space_quadtree_collides_aabb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                    quadtree_end = perf_counter_ns()
        else:  # OBB
            if split_method.lower() == "point":
                quadtree_start = perf_counter_ns()
                _ = qd.point_quadtree_collides_obb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                _ = qd.point_quadtree_collides_obb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                quadtree_end = perf_counter_ns()
            else:  # Space
                if compressed:
                    quadtree_start = perf_counter_ns()
                    _ = qd.compressed_quadtree_collides_obb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                    _ = qd.compressed_quadtree_collides_obb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                    quadtree_end = perf_counter_ns()
                else:
                    quadtree_start = perf_counter_ns()
                    _ = qd.space_quadtree_collides_obb(quadtree, arm1_bb)  # pyright: ignore[reportArgumentType]
                    _ = qd.space_quadtree_collides_obb(quadtree, arm2_bb)  # pyright: ignore[reportArgumentType]
                    quadtree_end = perf_counter_ns()

        rel_time = (quadtree_end - quadtree_start) / (naive_end - naive_start)
        speed_factor = 1 / rel_time
        results.append(
            TimeResults(naive_end - naive_start, quadtree_end - quadtree_start, speed_factor)
        )

    avg_naive_time = avg_quadtree_time = 0
    avg_factor = 0.0
    for naive_time, quadtree_time, factor in results:
        avg_naive_time += naive_time
        avg_quadtree_time += quadtree_time
        avg_factor += factor
    avg_naive_time /= sample_size
    avg_quadtree_time /= sample_size
    avg_factor /= sample_size

    faster = avg_factor >= 1
    if not faster:
        avg_factor = 1 / avg_factor

    print(fmt.bold("Average Performance of Collision"))
    print(
        fmt.it(f"{sample_size} samples of {num_points} points in ({world.width}, {world.height})")
    )
    if robot_override:
        print(
            f"{ANSI.DIM}Robot: (l: {robot.arm_width}, r2: {robot.arm1_length}, r3: {robot.arm2_length}){ANSI.RESET}"  # pyright: ignore[reportOptionalMemberAccess]
        )

    print(f"Naive approach: {avg_naive_time} ns")

    quadtree_str = f"Quadtree ({split_method.upper()}, {bb_type.upper()}"
    if compressed:
        quadtree_str += ", Compressed"
    quadtree_str += f"): {avg_quadtree_time} ns ({(avg_factor):.3f}x {fmt.pprint_bool(faster, ('faster', 'slower'))})"
    print(quadtree_str)
    fmt.LF()

    return (TimeResults(avg_naive_time, avg_quadtree_time, avg_factor), points)  # pyright: ignore[reportPossiblyUnboundVariable]


def compare_kdtree(
    world: World,
    sample_size: int,
    clusters: int,
    num_points: int,
    other: ComparisonConfig,
    build_times: bool,
) -> tuple[list[TimeResults], list[Point]]:
    if any(
        [
            sample_size <= 0,
            num_points <= 0,
            not isinstance(sample_size, int),
            not isinstance(num_points, int),
        ]
    ):
        raise ValueError("`sample_size` and `num_points` take positive integer values.")
    if other.structure not in {"naive", "quadtree"}:
        raise ValueError("`other` must be either 'naive' or 'quadtree'.")

    results: list[TimeResults] = []
    build_results: list[TimeResults] = []

    def build_time(fn: Callable, *args: object) -> tuple[int, object]:
        start = perf_counter_ns()
        struct = fn(*args)
        end = perf_counter_ns()
        return (end - start, struct)

    def perf_time(fn: Callable, *args: object) -> int:
        start = perf_counter_ns()
        _ = fn(*args)
        end = perf_counter_ns()
        return end - start

    def percentile(values: list[float], percent: float) -> float:
        ordered = sorted(values)
        index = (len(ordered) - 1) * percent / 100
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        if ordered[lower] == ordered[upper]:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    points: list[Point] = []
    for _ in range(sample_size):
        q_init = generate_rand_conf_prop_world(world)
        q_eval = generate_rand_conf_prop_world(world)
        points = world.generate_obstacles(
            clusters, num_points, 1, max(world.width, world.height) / 2, q_init
        )
        robot_envelope = make_aabb_from_circle(get_navigable_area(q_eval)[1])
        # --- Generate config bounding boxes
        if other.bb_type == "aabb":
            arm1_bb = get_link_aabb(q_eval, 0)
            arm2_bb = get_link_aabb(q_eval, 1)
        else:
            arm1_bb = get_link_obb(q_eval, 0)
            arm2_bb = get_link_obb(q_eval, 1)

        # --- Other ---
        if other.structure.lower() == "naive":
            if build_times:
                other_build_time = float("inf")
            other_time = perf_time(conf_is_valid_naive, q_eval, points)
        else:
            if other.split_method is None:
                other.split_method = "space"
            if other.bb_type is None:
                other.bb_type = "aabb"
            # --- Build structure ---
            if other.split_method == "space":
                if build_times:
                    other_build_time, quadtree = build_time(
                        qd.build_space_quadtree, points, robot_envelope, 1
                    )
                else:
                    quadtree = qd.build_space_quadtree(points, robot_envelope, 1)
                if other.compressed:
                    if build_times:
                        res = build_time(qd.compress_quadtree, quadtree)
                        other_build_time += res[0]  # pyright: ignore[reportPossiblyUnboundVariable]
                        quadtree = res[1]
                    else:
                        quadtree = qd.compress_quadtree(quadtree)  # pyright: ignore[reportArgumentType]
            else:
                if build_times:
                    other_build_time, quadtree = build_time(
                        qd.build_point_quadtree, points, robot_envelope, 1
                    )
                else:
                    quadtree = qd.build_point_quadtree(points, robot_envelope, 1)

            # --- Time other ---
            if other.split_method == "space":
                if other.bb_type == "aabb":
                    if other.compressed:
                        other_time = perf_time(
                            qd.compressed_quadtree_collides_aabb, quadtree, arm1_bb
                        ) + perf_time(qd.compressed_quadtree_collides_aabb, quadtree, arm2_bb)
                    else:
                        other_time = perf_time(
                            qd.space_quadtree_collides_aabb, quadtree, arm1_bb
                        ) + perf_time(qd.space_quadtree_collides_aabb, quadtree, arm2_bb)
                else:
                    if other.compressed:
                        other_time = perf_time(
                            qd.compressed_quadtree_collides_obb, quadtree, arm1_bb
                        ) + perf_time(qd.compressed_quadtree_collides_obb, quadtree, arm2_bb)
                    else:
                        other_time = perf_time(
                            qd.space_quadtree_collides_obb, quadtree, arm1_bb
                        ) + perf_time(qd.space_quadtree_collides_obb, quadtree, arm2_bb)
            else:  # point
                if other.bb_type == "aabb":
                    other_time = perf_time(
                        qd.point_quadtree_collides_aabb, quadtree, arm1_bb
                    ) + perf_time(qd.point_quadtree_collides_aabb, quadtree, arm2_bb)
                else:
                    other_time = perf_time(
                        qd.point_quadtree_collides_obb, quadtree, arm1_bb
                    ) + perf_time(qd.point_quadtree_collides_obb, quadtree, arm2_bb)

        # --- Kd-Tree ---
        if build_times:
            kdtree_build_time, kdtree = build_time(kd.build_kdtree, points, robot_envelope, 1)
        else:
            kdtree = kd.build_kdtree(points, robot_envelope, 1)
        kdtree_time = perf_time(kd.kdtree_collides_bb, kdtree, arm1_bb) + perf_time(
            kd.kdtree_collides_bb, kdtree, arm2_bb
        )

        speed_factor = other_time / kdtree_time
        if build_times:
            build_speed_factor = other_build_time / kdtree_build_time  # pyright: ignore[reportOperatorIssue, reportPossiblyUnboundVariable]
            build_results.append(  # pyright: ignore[reportOptionalMemberAccess]
                TimeResults(other_build_time, kdtree_build_time, build_speed_factor)  # pyright: ignore[reportPossiblyUnboundVariable]
            )
        results.append(TimeResults(other_time, kdtree_time, speed_factor))

    avg_other_time = avg_kdtree_time = avg_factor = 0.0
    for other_time, kdtree_time, factor in results:
        avg_other_time += other_time
        avg_kdtree_time += kdtree_time
        avg_factor += factor
    avg_other_time /= sample_size
    avg_kdtree_time /= sample_size
    avg_factor /= sample_size
    p99_other_time = percentile([result.first for result in results], 99)
    p99_kdtree_time = percentile([result.second for result in results], 99)
    p99_factor = p99_other_time / p99_kdtree_time

    faster = avg_factor >= 1
    if not faster:
        avg_factor = 1 / avg_factor

    p99_other_build_time = p99_kdtree_build_time = p99_build_factor = 0.0
    if build_times:
        avg_other_build_time = avg_kdtree_build_time = avg_build_factor = 0.0
        for other_build_time, kdtree_build_time, build_factor in build_results:
            avg_other_build_time += other_build_time
            avg_kdtree_build_time += kdtree_build_time
            avg_build_factor += build_factor
        avg_other_build_time /= sample_size
        avg_kdtree_build_time /= sample_size
        avg_build_factor /= sample_size
        p99_other_build_time = percentile([result.first for result in build_results], 99)
        p99_kdtree_build_time = percentile([result.second for result in build_results], 99)
        p99_build_factor = p99_other_build_time / p99_kdtree_build_time

        build_faster = avg_build_factor >= 1
        if not build_faster:
            avg_build_factor = 1 / avg_build_factor

    print(f"{ANSI.DIM}{fmt.bold('=' * 30)}{ANSI.RESET}")
    if build_times:
        print(fmt.bold("Average Build Time"))
        print(
            fmt.it(
                f"{sample_size} samples of {clusters * num_points} points in ({world.width}, {world.height})"
            )
        )
        other_str = f"{other.structure}"
        if other.structure == "quadtree":
            other_str += f" ({other.split_method}, {other.bb_type}"
            if other.compressed:
                other_str += ", compressed"
            other_str += ")"
        other_str += f": {avg_other_build_time:.0f} ns"  # pyright: ignore[reportPossiblyUnboundVariable]
        print(other_str)
        print(
            f"Kd-Tree: {avg_kdtree_build_time:.0f} ns ({avg_build_factor:.3f}x {  # pyright: ignore[reportPossiblyUnboundVariable]
                fmt.pprint_bool(build_faster, ('faster', 'slower'))  # pyright: ignore[reportPossiblyUnboundVariable]
            })"
        )
        print(fmt.it("--- 99th Percentile ---"))
        print(f"{other_str.split(': ')[0]}: {p99_other_build_time:.0f} ns")
        print(
            f"Kd-Tree: {p99_kdtree_build_time:.0f} ns ({p99_build_factor:.3f}x {  # pyright: ignore[reportPossiblyUnboundVariable]
                fmt.pprint_bool(p99_build_factor >= 1, ('faster', 'slower'))
            })"
        )
        fmt.LF()

    print(fmt.bold("Average Performance of Collision"))
    print(
        fmt.it(
            f"{sample_size} samples of {clusters * num_points} points in ({world.width}, {world.height})"
        )
    )
    other_str = f"{other.structure}"
    if other.structure == "quadtree":
        other_str += f" ({other.split_method}, {other.bb_type}"
        if other.compressed:
            other_str += ", compressed"
        other_str += ")"
    other_str += f": {avg_other_time:.0f} ns"
    print(other_str)
    print(
        f"Kd-Tree: {avg_kdtree_time:.0f} ns ({avg_factor:.3f}x {
            fmt.pprint_bool(faster, ('faster', 'slower'))
        })"
    )
    print(fmt.it("--- 99th Percentile ---"))
    print(f"{other_str.split(': ')[0]}: {p99_other_time:.0f} ns")
    print(
        f"Kd-Tree: {p99_kdtree_time:.0f} ns ({p99_factor:.3f}x {
            fmt.pprint_bool(p99_factor >= 1, ('faster', 'slower'))
        })"
    )
    print(f"{ANSI.DIM}{fmt.bold('=' * 30)}{ANSI.RESET}\n")

    results = [
        TimeResults(avg_other_time, avg_kdtree_time, avg_factor),
        TimeResults(p99_other_time, p99_kdtree_time, p99_factor),
    ]
    if build_times:
        results.append(TimeResults(avg_other_build_time, avg_kdtree_build_time, avg_build_factor))  # pyright: ignore[reportPossiblyUnboundVariable]
        results.append(TimeResults(p99_other_build_time, p99_kdtree_build_time, p99_build_factor))

    return (results, points)


def main() -> None:
    world = World(3, 3)
    big_world = World(50, 50)
    q_init = generate_rand_conf_prop_world(world)
    points = world.generate_obstacles(5, 10, 1, max(world.width, world.height) / 2, q_init)

    # small_world = World(3, 3)
    # _ = compare_quadtree_to_naive(small_world, 1000, 100, "AABB", "space")
    # _ = compare_quadtree_to_naive(world, 1000, 100, "AABB", "space")
    # _ = compare_quadtree_to_naive(small_world, 1000, 10, "OBB", "space")
    # _ = compare_quadtree_to_naive(small_world, 1000, 100, "OBB", "space")
    # _ = compare_quadtree_to_naive(world, 1000, 100, "OBB", "space")
    # _ = compare_quadtree_to_naive(world, 1000, 720, "OBB", "space")
    # _ = compare_quadtree_to_naive(small_world, 1000, 100, "AABB", "point")
    # _ = compare_quadtree_to_naive(world, 1000, 100, "AABB", "point")
    # _ = compare_quadtree_to_naive(small_world, 1000, 10, "OBB", "point")
    # _ = compare_quadtree_to_naive(small_world, 1000, 100, "OBB", "point")
    # _ = compare_quadtree_to_naive(world, 1000, 100, "OBB", "point")
    # _ = compare_quadtree_to_naive(world, 1000, 400, "OBB", "point")
    # _ = compare_quadtree_to_naive(world, 1000, 100, "AABB", "space", True)
    # _ = compare_quadtree_to_naive(world, 1000, 100, "OBB", "space", True)

    # fmt: off
    # print(fmt.bold("=== World similar to assignment ===\n"))
    # results, _ = compare_kdtree(world, 1000, 10, 10, ComparisonConfig("naive"), True)
    # _ = plot_time_results(results[:2], "Kd-Tree VS Naive")
    # results, _ = compare_kdtree(world, 1000, 10, 10, ComparisonConfig("quadtree", "space", "aabb", False), True)
    # _ = plot_time_results(results, "Kd-Tree VS Space Quadtree (AABB)")

    # results, _ = compare_kdtree(world, 1000, 10, 10, ComparisonConfig("quadtree", "space", "aabb", True), True)
    # _ = plot_time_results(results, "Kd-Tree VS Space Quadtree (AABB, Compressed)")
    # results, _ = compare_kdtree(world, 1000, 10, 10, ComparisonConfig("quadtree", "space", "obb", False), True)
    # _ = plot_time_results(results, "Kd-Tree VS Space Quadtree (OBB)")
    # results, _ = compare_kdtree(world, 1000, 10, 10, ComparisonConfig("quadtree", "space", "obb", True), True)
    # _ = plot_time_results(results, "Kd-Tree VS Space Quadtree (OBB, Compressed)")
    # results, _ = compare_kdtree(world, 1000, 10, 10, ComparisonConfig("quadtree", "point", "aabb"), True)
    # _ = plot_time_results(results, "Kd-Tree VS Point Quadtree (AABB)")
    # results, _ = compare_kdtree(world, 1000, 10, 10, ComparisonConfig("quadtree", "point", "obb"), True)
    # _ = plot_time_results(results, "Kd-Tree VS Point Quadtree (OBB)")

    print(fmt.bold("=== 'Sparse' world ===\n"))
    _ = compare_kdtree(big_world, 1000, 10, 10, ComparisonConfig("naive"), True)
    _ = compare_kdtree(big_world, 1000, 10, 10, ComparisonConfig("quadtree", "space", "aabb", False), True)
    _ = compare_kdtree(big_world, 1000, 10, 10, ComparisonConfig("quadtree", "space", "aabb", True), True)
    _ = compare_kdtree(big_world, 1000, 10, 10, ComparisonConfig("quadtree", "space", "obb", False), True)
    _ = compare_kdtree(big_world, 1000, 10, 10, ComparisonConfig("quadtree", "space", "obb", True), True)
    _ = compare_kdtree(big_world, 1000, 10, 10, ComparisonConfig("quadtree", "point", "aabb"), True)
    _ = compare_kdtree(big_world, 1000, 10, 10, ComparisonConfig("quadtree", "point", "obb"), True), True

    # print(fmt.bold("=== World similar to LiDAR measures ===\n"))
    # _ = compare_kdtree(big_world, 1000, 30, 30, ComparisonConfig("naive"), True)
    # _ = compare_kdtree(big_world, 1000, 30, 30, ComparisonConfig("quadtree", "space", "aabb", False), True)
    # _ = compare_kdtree(big_world, 1000, 30, 30, ComparisonConfig("quadtree", "space", "aabb", True), True)
    # _ = compare_kdtree(big_world, 1000, 30, 30, ComparisonConfig("quadtree", "space", "obb", False), True)
    # _ = compare_kdtree(big_world, 1000, 30, 30, ComparisonConfig("quadtree", "space", "obb", True), True)
    # _ = compare_kdtree(big_world, 1000, 30, 30, ComparisonConfig("quadtree", "point", "aabb"), True)
    # _ = compare_kdtree(big_world, 1000, 30, 30, ComparisonConfig("quadtree", "point", "obb"), True)
    # fmt: on

    robot_conf = generate_rand_conf_prop_world(world)
    navigable_area = get_navigable_area(robot_conf)
    # Micro-optimization: build the quadtree from the robot's envelope to prune unreachable points
    quadtree_region = make_aabb_from_circle(navigable_area[1])
    arm1_bb = get_link_aabb(robot_conf, 0)
    arm2_bb = get_link_aabb(robot_conf, 1)
    arm1_obb = get_link_obb(robot_conf, 0)
    arm2_obb = get_link_obb(robot_conf, 1)

    print(fmt.bold("Configuration is valid ?"))
    quadtree = qd.build_space_quadtree(points, quadtree_region, capacity=2)
    # quadtree = qd.compress_quadtree(quadtree)
    conf_is_valid_aabb = not qd.space_quadtree_collides_aabb(quadtree, arm1_bb)
    conf_is_valid_aabb &= not qd.space_quadtree_collides_aabb(quadtree, arm2_bb)
    conf_is_valid_obb = not qd.space_quadtree_collides_obb(quadtree, arm1_obb)
    conf_is_valid_obb &= not qd.space_quadtree_collides_obb(quadtree, arm2_obb)
    print(f"\t(Space, AABB): {fmt.pprint_bool(conf_is_valid_aabb)}")
    print(f"\t(Space, OBB): {fmt.pprint_bool(conf_is_valid_obb)}")
    fmt.LF()

    quadtree = qd.build_point_quadtree(points, quadtree_region, capacity=2)
    conf_is_valid_aabb = not qd.point_quadtree_collides_aabb(quadtree, arm1_bb)
    conf_is_valid_aabb &= not qd.point_quadtree_collides_aabb(quadtree, arm2_bb)
    conf_is_valid_obb = not qd.point_quadtree_collides_obb(quadtree, arm1_obb)
    conf_is_valid_obb &= not qd.point_quadtree_collides_obb(quadtree, arm2_obb)
    print(f"\t(Point, AABB): {fmt.pprint_bool(conf_is_valid_aabb)}")
    print(f"\t(Point, OBB): {fmt.pprint_bool(conf_is_valid_obb)}")
    fmt.LF()

    kdtree = kd.build_kdtree(points, quadtree_region, capacity=2)
    conf_is_valid_aabb = not kd.kdtree_collides_bb(kdtree, arm1_bb)
    conf_is_valid_aabb &= not kd.kdtree_collides_bb(kdtree, arm2_bb)
    conf_is_valid_obb = not kd.kdtree_collides_bb(kdtree, arm1_obb)
    conf_is_valid_obb &= not kd.kdtree_collides_bb(kdtree, arm2_obb)
    print(f"\t(Kd-Tree, AABB): {fmt.pprint_bool(conf_is_valid_aabb)}")
    print(f"\t(Kd-Tree, OBB): {fmt.pprint_bool(conf_is_valid_obb)}")
    fmt.LF()

    _ = renderer.draw_world(world)
    _ = renderer.draw_robot_config(robot_conf)
    _ = renderer.draw_quadtree(quadtree)
    # _ = renderer.draw_compressed_quadtree(quadtree)
    # _ = renderer.draw_kdtree(kdtree)
    _ = renderer.draw_points(points, 100)
    # _ = renderer.draw_navigable_area(navigable_area)
    _ = renderer.draw_aabb(arm1_bb, (0.5, 0, 0.5))
    _ = renderer.draw_aabb(arm2_bb, (0.5, 0.5, 0))
    _ = renderer.draw_obb(arm1_obb, (0, 0, 0.5))
    _ = renderer.draw_obb(arm2_obb, (0.5, 0, 0))

    plt.show(block=True)


if __name__ == "__main__":
    main()
