"""
Plain RRT-Connect solver -- no visualization, just the algorithm.

    Put Start node into Start Tree
    Put Goal node into Goal Tree
    LOOP
        Sample random state
        Extend Tree A by one step
        Aggressively connect Tree B
        Connected?  Yes -> build final path  /  No -> continue
        Swap trees
    Return solution

2D Euclidean points, straight-line steering, obstacle hardcoded below.
For the SE(2) / Reeds-Shepp-steered version, see
rrt_connect_reeds_shepp_interactive.py.

Usage:
    python rrt_connect_solver.py --start 1 1 --goal 9 9
    python rrt_connect_solver.py --start 1 1 --goal 9 9 --step-size 1.5 --seed 42
"""

import argparse
import math
import random

# ----------------------------------------------------------------------
# World -- edit these to change the map. CLI only controls start/goal.
# ----------------------------------------------------------------------
BOUNDS = (0.0, 10.0)
OBSTACLE = dict(x=5.0, y=4.0, w=2.0, h=2.0)

DEFAULT_STEP_SIZE = 1.0
DEFAULT_MAX_ITER = 5000
DEFAULT_GOAL_TOLERANCE = 1e-6  # CONNECT is exact-Reached-or-not; kept for float safety


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def obstacle_contains(p):
    x, y = p
    return (OBSTACLE["x"] <= x <= OBSTACLE["x"] + OBSTACLE["w"] and
            OBSTACLE["y"] <= y <= OBSTACLE["y"] + OBSTACLE["h"])


def collision_free(p1, p2, samples=50):
    for i in range(samples + 1):
        t = i / samples
        p = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        if obstacle_contains(p):
            return False
    return True


def steer(p_from, p_to, step_size):
    d = dist(p_from, p_to)
    if d <= step_size:
        return p_to, True  # (new point, reached-exactly)
    ux, uy = (p_to[0] - p_from[0]) / d, (p_to[1] - p_from[1]) / d
    return (p_from[0] + ux * step_size, p_from[1] + uy * step_size), False


class Tree:
    def __init__(self, root):
        self.points = [root]
        self.parents = [None]

    def nearest(self, p):
        return min(range(len(self.points)), key=lambda i: dist(self.points[i], p))

    def add(self, p, parent_i):
        self.points.append(p)
        self.parents.append(parent_i)
        return len(self.points) - 1

    def path_to_root(self, i):
        path = []
        while i is not None:
            path.append(self.points[i])
            i = self.parents[i]
        path.reverse()
        return path


def extend(tree, target, step_size):
    """One EXTEND() step. Returns ('Reached'|'Advanced'|'Trapped', node_index_or_None)."""
    near_i = tree.nearest(target)
    p_near = tree.points[near_i]
    p_new, reached_exactly = steer(p_near, target, step_size)
    if not collision_free(p_near, p_new):
        return "Trapped", None
    new_i = tree.add(p_new, near_i)
    return ("Reached" if reached_exactly else "Advanced"), new_i


def connect(tree, target, step_size, max_steps=1000):
    """CONNECT(): repeat EXTEND until it doesn't just Advance."""
    status, node_i = "Advanced", None
    steps = 0
    while status == "Advanced":
        steps += 1
        if steps > max_steps:
            return "Trapped", None
        status, node_i = extend(tree, target, step_size)
    return status, node_i


def solve(start, goal, step_size=DEFAULT_STEP_SIZE, max_iter=DEFAULT_MAX_ITER, seed=None):
    """Run RRT-Connect from `start` to `goal` (both (x, y) tuples).
    Returns a list of (x, y) points from start to goal, or None if no
    solution was found within max_iter iterations."""
    if not collision_free(start, start) or not collision_free(goal, goal):
        raise ValueError("start or goal lies inside the obstacle")

    rng = random.Random(seed)
    tree_start = Tree(start)
    tree_goal = Tree(goal)
    Ta, Tb = tree_start, tree_goal

    for _ in range(max_iter):
        q_rand = (rng.uniform(*BOUNDS), rng.uniform(*BOUNDS))

        status_a, new_i_a = extend(Ta, q_rand, step_size)
        if status_a != "Trapped":
            q_new = Ta.points[new_i_a]
            status_b, new_i_b = connect(Tb, q_new, step_size)
            if status_b == "Reached":
                path_a = Ta.path_to_root(new_i_a)          # Ta.root -> meeting
                path_b = Tb.path_to_root(new_i_b)           # Tb.root -> meeting
                combined = path_a + list(reversed(path_b))[1:]
                return combined if Ta.points[0] == start else list(reversed(combined))

        Ta, Tb = Tb, Ta

    return None


def parse_point(vals):
    return (float(vals[0]), float(vals[1]))


def main():
    parser = argparse.ArgumentParser(description="Plain RRT-Connect solver (no visualization)")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    start = parse_point(args.start)
    goal = parse_point(args.goal)

    try:
        path = solve(start, goal, step_size=args.step_size, max_iter=args.max_iter, seed=args.seed)
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    if path is None:
        print("No path found.")
        return

    print(f"Path found with {len(path)} point(s):")
    for p in path:
        print(f"  ({p[0]:.3f}, {p[1]:.3f})")
    total_len = sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    print(f"Total length: {total_len:.3f}")


if __name__ == "__main__":
    main()
