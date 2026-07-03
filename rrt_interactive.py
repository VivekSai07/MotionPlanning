"""
Interactive RRT (Rapidly-exploring Random Tree) step-through visualizer.

Pops up a matplotlib window with:
  - Left panel  : the world (obstacle, tree, current random sample / steer /
                  collision check), updated one micro-step at a time.
  - Upper-right : the algorithm pseudocode, with the line currently being
                  executed highlighted.
  - Lower-right : "Function Detail" -- the actual arithmetic happening
                  inside RandomSample() / Nearest() / Steer() / CollisionFree()
                  for the current step (distances to every node, the
                  direction-vector math, which sample point along the
                  candidate edge triggered a collision, etc).
  - Buttons     : "<< Back" / "Next >>" / "Reset (new run)" / a sampling-mode
                  toggle -- click through the algorithm forwards or
                  backwards at your own pace.
                  (Left/Right arrow keys and 'r' work too.)

Random points are genuinely generated with Python's `random` module (a new
sequence every run, or pass --seed for a reproducible one). Sampling is
pure uniform by default; the "Sampling" button toggles a goal-bias mode
(sample the goal directly some % of the time) on/off at any point, even
mid-run -- goal-biasing is a standard, well documented RRT enhancement,
not a scripted path.

Run:
    python rrt_interactive.py
    python rrt_interactive.py --seed 42            # reproducible run
    python rrt_interactive.py --goal-bias 0.10     # start with goal-bias on
"""

import argparse
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

# ----------------------------------------------------------------------
# World setup
# ----------------------------------------------------------------------
START = (1.0, 1.0)
GOAL = (9.0, 9.0)
BOUNDS = (0.0, 10.0)
STEP_SIZE = 2.0
GOAL_THRESHOLD = 1.0
OBSTACLE = dict(x=5.0, y=4.0, w=2.0, h=2.0)
MAX_ITERATIONS = 500
COLLISION_SAMPLES = 200
COLLISION_VIZ_STRIDE = 5   # draw every Nth sample point tested along an edge
DEFAULT_GOAL_BIAS_ON = 0.10

CODE_LINES = [
    "Initialize Tree with start node",
    "while goal not reached:",
    "    q_rand = RandomSample()",
    "    q_near = Nearest(Tree, q_rand)",
    "    q_new  = Steer(q_near, q_rand)",
    "    if CollisionFree(q_near, q_new):",
    "        Add q_new to Tree",
    "return path",
]


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def obstacle_contains(p, margin=0.0):
    x, y = p
    return (OBSTACLE["x"] - margin <= x <= OBSTACLE["x"] + OBSTACLE["w"] + margin and
            OBSTACLE["y"] - margin <= y <= OBSTACLE["y"] + OBSTACLE["h"] + margin)


def collision_check_detailed(p1, p2, samples=COLLISION_SAMPLES, viz_stride=COLLISION_VIZ_STRIDE):
    """Sample the segment p1->p2 densely and report exactly what CollisionFree()
    is doing: which points were tested, and (if any) the first one that landed
    inside the obstacle. Returns (ok, first_collision, viz_points).
    """
    viz_points = []
    first_collision = None
    for i in range(samples + 1):
        t = i / samples
        p = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        inside = obstacle_contains(p)
        if inside and first_collision is None:
            first_collision = (t, p)
        if i % viz_stride == 0 or inside:
            viz_points.append((p, inside))
        if inside:
            break  # a real collision check can stop at the first hit
    ok = first_collision is None
    return ok, first_collision, viz_points


def nearest_node_detailed(nodes, q_rand):
    """Real nearest-neighbor search, returning the full per-node distance
    table (what Nearest() actually computes) alongside the winner."""
    dists = [dist(n["pos"], q_rand) for n in nodes]
    near_i = min(range(len(nodes)), key=lambda i: dists[i])
    return near_i, dists[near_i], dists


def steer_detailed(q_near, q_rand, step):
    dx = q_rand[0] - q_near[0]
    dy = q_rand[1] - q_near[1]
    d = math.hypot(dx, dy)
    if d <= step:
        return q_rand, dict(dx=dx, dy=dy, d=d, truncated=False)
    ux, uy = dx / d, dy / d
    sx, sy = ux * step, uy * step
    q_new = (q_near[0] + sx, q_near[1] + sy)
    return q_new, dict(dx=dx, dy=dy, d=d, truncated=True, ux=ux, uy=uy, sx=sx, sy=sy)


# ----------------------------------------------------------------------
# RRT engine: a generator that yields one micro-step (a "snapshot") at a
# time, so the UI can drive it forward on demand. Nothing is precomputed
# in bulk -- each call to next() performs real, fresh random sampling, and
# self.goal_bias is read fresh every iteration so the UI can toggle
# sampling mode mid-run.
# ----------------------------------------------------------------------
class RRTStepper:
    def __init__(self, rng, goal_bias):
        self.rng = rng
        self.goal_bias = goal_bias
        self.nodes = [{"pos": START, "parent": None}]
        self.goal_reached = False
        self.path = None

    def _tree_edges(self):
        return [(self.nodes[n["parent"]]["pos"], n["pos"])
                for n in self.nodes if n["parent"] is not None]

    def _tree_nodes(self):
        return [n["pos"] for n in self.nodes]

    def _base_snapshot(self, **kw):
        snap = dict(
            tree_nodes=list(self._tree_nodes()),
            tree_edges=list(self._tree_edges()),
            title="",
            info="",
            info_color="black",
            detail="",
        )
        snap.update(kw)
        return snap

    def run(self):
        yield self._base_snapshot(
            type="init", highlight=0,
            title="Initialize",
            info=f"Tree = {{ start = {START} }}",
            detail=f"Tree initialized with a single node:\n  start = {START}",
        )

        iteration = 0
        while not self.goal_reached and iteration < MAX_ITERATIONS:
            iteration += 1
            yield self._base_snapshot(
                type="loop", highlight=1,
                title=f"Iteration {iteration}: goal not yet reached, continue",
                info=f"Tree has {len(self.nodes)} node(s) so far.",
                detail=(f"while goal not reached:\n"
                        f"  tree currently has {len(self.nodes)} node(s)\n"
                        f"  goal not yet in tree -> enter loop body"),
            )

            # --- RandomSample() ---
            biased = self.rng.random() < self.goal_bias
            if biased:
                q_rand = GOAL
                sample_detail = (f"RandomSample(): goal-bias roll succeeded\n"
                                  f"  (probability = {self.goal_bias:.0%})\n"
                                  f"  q_rand = Goal = {GOAL}")
            else:
                q_rand = (self.rng.uniform(*BOUNDS), self.rng.uniform(*BOUNDS))
                mode_note = "  (goal-bias roll failed)" if self.goal_bias > 0 else ""
                sample_detail = (f"RandomSample(): uniform draw over map bounds{mode_note}\n"
                                  f"  x ~ Uniform{BOUNDS} = {q_rand[0]:.3f}\n"
                                  f"  y ~ Uniform{BOUNDS} = {q_rand[1]:.3f}\n"
                                  f"  q_rand = ({q_rand[0]:.3f}, {q_rand[1]:.3f})")
            yield self._base_snapshot(
                type="sample", highlight=2, q_rand=q_rand,
                title=f"Iteration {iteration}: Random Sample",
                info=(f"q_rand = ({q_rand[0]:.2f}, {q_rand[1]:.2f})"
                      + ("  [goal-biased sample]" if biased else "")),
                info_color="darkorange",
                detail=sample_detail,
            )

            # --- Nearest(Tree, q_rand) ---
            near_i, near_d, all_dists = nearest_node_detailed(self.nodes, q_rand)
            q_near = self.nodes[near_i]["pos"]
            order = sorted(range(len(all_dists)), key=lambda i: all_dists[i])
            show_n = min(6, len(order))
            lines = [f"Nearest(Tree, q_rand): Euclidean distance to each of {len(all_dists)} node(s)"]
            for idx in order[:show_n]:
                p = self.nodes[idx]["pos"]
                marker = "  <- NEAREST" if idx == near_i else ""
                lines.append(f"  node[{idx}] ({p[0]:.2f},{p[1]:.2f})  dist={all_dists[idx]:.2f}{marker}")
            if len(order) > show_n:
                lines.append(f"  ... and {len(order) - show_n} more node(s)")
            lines.append(f"  => nearest = node[{near_i}], dist={near_d:.2f}")
            yield self._base_snapshot(
                type="nearest", highlight=3, q_rand=q_rand, q_near=q_near, all_dists=all_dists,
                title=f"Iteration {iteration}: Nearest Node",
                info=f"q_near = ({q_near[0]:.2f}, {q_near[1]:.2f})   dist = {near_d:.2f}",
                info_color="blue",
                detail="\n".join(lines),
            )

            # --- Steer(q_near, q_rand) ---
            q_new, sd = steer_detailed(q_near, q_rand, STEP_SIZE)
            if sd["truncated"]:
                steer_detail = (
                    f"Steer(q_near, q_rand, step={STEP_SIZE}):\n"
                    f"  dx = {q_rand[0]:.2f} - {q_near[0]:.2f} = {sd['dx']:.2f}\n"
                    f"  dy = {q_rand[1]:.2f} - {q_near[1]:.2f} = {sd['dy']:.2f}\n"
                    f"  |d| = sqrt(dx^2+dy^2) = {sd['d']:.2f}  (> step, so truncate)\n"
                    f"  unit = (dx/|d|, dy/|d|) = ({sd['ux']:.3f}, {sd['uy']:.3f})\n"
                    f"  step_vec = step * unit = ({sd['sx']:.3f}, {sd['sy']:.3f})\n"
                    f"  q_new = q_near + step_vec = ({q_new[0]:.2f}, {q_new[1]:.2f})"
                )
            else:
                steer_detail = (
                    f"Steer(q_near, q_rand, step={STEP_SIZE}):\n"
                    f"  |d| = {sd['d']:.2f}  (<= step, no truncation needed)\n"
                    f"  q_new = q_rand = ({q_new[0]:.2f}, {q_new[1]:.2f})"
                )
            yield self._base_snapshot(
                type="steer", highlight=4, q_rand=q_rand, q_near=q_near, q_new=q_new,
                title=f"Iteration {iteration}: Steer (Δq = {STEP_SIZE})",
                info=f"q_new = ({q_new[0]:.2f}, {q_new[1]:.2f})",
                info_color="black",
                detail=steer_detail,
            )

            # --- CollisionFree(q_near, q_new) ---
            ok, first_collision, viz_points = collision_check_detailed(q_near, q_new)
            ox0, oy0 = OBSTACLE["x"], OBSTACLE["y"]
            ox1, oy1 = ox0 + OBSTACLE["w"], oy0 + OBSTACLE["h"]
            if ok:
                coll_detail = (
                    f"CollisionFree(q_near, q_new): tested {COLLISION_SAMPLES + 1} points along the segment\n"
                    f"  obstacle = x:[{ox0:.0f},{ox1:.0f}], y:[{oy0:.0f},{oy1:.0f}]\n"
                    f"  every sample point lies outside the obstacle\n"
                    f"  => RESULT: collision-free, edge accepted"
                )
            else:
                t, p = first_collision
                coll_detail = (
                    f"CollisionFree(q_near, q_new): tested points along the segment\n"
                    f"  obstacle = x:[{ox0:.0f},{ox1:.0f}], y:[{oy0:.0f},{oy1:.0f}]\n"
                    f"  first hit at t={t:.2f} -> point=({p[0]:.2f},{p[1]:.2f}) is INSIDE the obstacle\n"
                    f"  => RESULT: collision detected, edge discarded"
                )
            yield self._base_snapshot(
                type="collision", highlight=5, q_near=q_near, q_new=q_new, ok=ok,
                viz_points=viz_points,
                title=f"Iteration {iteration}: Collision Check",
                info=("Edge is collision-free" if ok else "Edge COLLIDES with obstacle -> discard"),
                info_color=("green" if ok else "red"),
                detail=coll_detail,
            )

            if not ok:
                continue  # discarded; back to top of while loop

            # --- Add q_new ---
            self.nodes.append({"pos": q_new, "parent": near_i})
            new_i = len(self.nodes) - 1
            yield self._base_snapshot(
                type="add", highlight=6, q_new=q_new,
                title=f"Iteration {iteration}: Add Node",
                info=f"Added ({q_new[0]:.2f}, {q_new[1]:.2f}) to the tree.",
                info_color="darkgreen",
                detail=(f"Add q_new to Tree:\n"
                        f"  new node[{new_i}] = ({q_new[0]:.2f}, {q_new[1]:.2f})\n"
                        f"  parent = node[{near_i}] ({q_near[0]:.2f}, {q_near[1]:.2f})\n"
                        f"  tree size now = {len(self.nodes)}"),
            )

            # --- Goal check (part of the while-condition) ---
            d_goal = dist(q_new, GOAL)
            if d_goal < GOAL_THRESHOLD:
                yield self._base_snapshot(
                    type="goalcheck", highlight=1, q_new=q_new, d=d_goal,
                    title="Goal Check",
                    info=f"dist(q_new, Goal) = {d_goal:.2f} < threshold={GOAL_THRESHOLD} -> try final connection",
                    info_color="purple",
                    detail=(f"Check: dist(q_new, Goal) < threshold?\n"
                            f"  dist = {d_goal:.2f}, threshold = {GOAL_THRESHOLD}\n"
                            f"  condition TRUE -> attempt final CollisionFree(q_new, Goal)"),
                )
                ok_goal, _, viz_points_g = collision_check_detailed(q_new, GOAL)
                if ok_goal:
                    self.nodes.append({"pos": GOAL, "parent": new_i})
                    self.goal_reached = True
                    goal_i = len(self.nodes) - 1
                    path = []
                    i = goal_i
                    while i is not None:
                        path.append(self.nodes[i]["pos"])
                        i = self.nodes[i]["parent"]
                    path.reverse()
                    self.path = path
                    yield self._base_snapshot(
                        type="connect", highlight=1, q_new=q_new, goal=GOAL, viz_points=viz_points_g,
                        title="Goal Reached!",
                        info="Final edge to goal is collision-free -> connected.",
                        info_color="darkgreen",
                        detail="CollisionFree(q_new, Goal) -> collision-free\n  Goal connected to tree!",
                    )
                    yield self._base_snapshot(
                        type="path", highlight=7, path=path,
                        title="Return Path",
                        info=" -> ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in path),
                        info_color="magenta",
                        detail=("return path: follow parent pointers from Goal back to Start\n"
                                f"  path length = {len(path)} node(s)\n"
                                "  (see highlighted magenta path)"),
                    )
                # if the final connection collides, the node stays in the
                # tree but we simply keep looping -- exactly like the base
                # algorithm, no special-casing needed.

        if not self.goal_reached:
            yield self._base_snapshot(
                type="failed", highlight=1,
                title="No path found",
                info=f"Reached MAX_ITERATIONS={MAX_ITERATIONS} without connecting to the goal.",
                info_color="red",
                detail=f"Iteration limit reached ({MAX_ITERATIONS}).\n  No path found in this run.",
            )


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
class App:
    def __init__(self, seed, goal_bias):
        self.seed = seed
        self.goal_bias = goal_bias
        self.goal_bias_on_value = goal_bias if goal_bias > 0 else DEFAULT_GOAL_BIAS_ON
        self.history = []
        self.cursor = -1
        self.stepper = None
        self._new_run()

        self.fig = plt.figure(figsize=(14.5, 8))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.1, 1.0], height_ratios=[1.0, 1.3],
                                    left=0.05, right=0.97, top=0.93, bottom=0.16,
                                    wspace=0.15, hspace=0.25)
        self.ax_world = self.fig.add_subplot(gs[:, 0])
        self.ax_code = self.fig.add_subplot(gs[0, 1])
        self.ax_detail = self.fig.add_subplot(gs[1, 1])

        ax_back = self.fig.add_axes([0.16, 0.03, 0.13, 0.06])
        ax_next = self.fig.add_axes([0.31, 0.03, 0.13, 0.06])
        ax_reset = self.fig.add_axes([0.46, 0.03, 0.19, 0.06])
        ax_mode = self.fig.add_axes([0.67, 0.03, 0.27, 0.06])
        self.btn_back = Button(ax_back, "<< Back")
        self.btn_next = Button(ax_next, "Next >>")
        self.btn_reset = Button(ax_reset, "Reset (new run)")
        self.btn_mode = Button(ax_mode, "")
        self._update_mode_button_label()
        self.btn_back.on_clicked(self.on_back)
        self.btn_next.on_clicked(self.on_next)
        self.btn_reset.on_clicked(self.on_reset)
        self.btn_mode.on_clicked(self.on_toggle_mode)

        self.status_text = self.fig.text(0.05, 0.11, "", fontsize=9, family="monospace")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.on_next(None)  # render the initial snapshot

    def _new_run(self):
        actual_seed = self.seed if self.seed is not None else random.SystemRandom().randrange(2**31)
        print(f"Starting new run with seed={actual_seed}, goal_bias={self.goal_bias}")
        rng = random.Random(actual_seed)
        self.stepper = RRTStepper(rng, self.goal_bias)
        self.gen = self.stepper.run()
        self.history = []
        self.cursor = -1

    # -- sampling mode ----------------------------------------------------
    def _update_mode_button_label(self):
        label = ("Sampling: Goal-biased (%.0f%%)" % (self.goal_bias * 100)
                  if self.goal_bias > 0 else "Sampling: Uniform")
        self.btn_mode.label.set_text(label)

    def on_toggle_mode(self, event):
        self.goal_bias = 0.0 if self.goal_bias > 0 else self.goal_bias_on_value
        if self.stepper is not None:
            self.stepper.goal_bias = self.goal_bias  # takes effect immediately, even mid-run
        self._update_mode_button_label()
        self.fig.canvas.draw_idle()

    # -- navigation -----------------------------------------------------
    def on_next(self, event):
        if self.cursor < len(self.history) - 1:
            self.cursor += 1
        else:
            try:
                snap = next(self.gen)
            except StopIteration:
                return
            self.history.append(snap)
            self.cursor += 1
        self.render()

    def on_back(self, event):
        if self.cursor > 0:
            self.cursor -= 1
            self.render()

    def on_reset(self, event):
        self._new_run()
        self.on_next(None)

    def on_key(self, event):
        if event.key == "right":
            self.on_next(None)
        elif event.key == "left":
            self.on_back(None)
        elif event.key == "r":
            self.on_reset(None)

    # -- drawing ----------------------------------------------------------
    def _draw_static_world(self):
        ax = self.ax_world
        ax.set_xlim(*BOUNDS)
        ax.set_ylim(*BOUNDS)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, 11))
        ax.set_yticks(range(0, 11))
        ax.grid(True, linestyle=":", alpha=0.4)
        rect = patches.Rectangle((OBSTACLE["x"], OBSTACLE["y"]), OBSTACLE["w"], OBSTACLE["h"],
                                  facecolor="0.25", edgecolor="black", zorder=2)
        ax.add_patch(rect)
        ax.plot(*START, marker="*", color="green", markersize=18, zorder=5)
        ax.annotate("Start", START, textcoords="offset points", xytext=(8, -14))
        ax.plot(*GOAL, marker="*", color="red", markersize=18, zorder=5)
        ax.annotate("Goal", GOAL, textcoords="offset points", xytext=(-30, 8))

    def render(self):
        snap = self.history[self.cursor]
        ax = self.ax_world
        ax.clear()
        self._draw_static_world()

        for (p1, p2) in snap["tree_edges"]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="steelblue", linewidth=2, zorder=3)
        for p in snap["tree_nodes"]:
            ax.plot(*p, "o", color="steelblue", markersize=6, zorder=4)

        t = snap["type"]
        if t == "sample":
            ax.plot(*snap["q_rand"], "x", color="darkorange", markersize=14, mew=3, zorder=6)
        elif t == "nearest":
            # faint line + distance-so-far from q_rand to EVERY node, so you
            # can see exactly what Nearest() is comparing.
            for p in snap["tree_nodes"]:
                ax.plot([p[0], snap["q_rand"][0]], [p[1], snap["q_rand"][1]],
                        "-", color="gray", alpha=0.25, linewidth=1, zorder=2)
            ax.plot(*snap["q_rand"], "x", color="darkorange", markersize=14, mew=3, zorder=6)
            ax.plot(*snap["q_near"], "o", color="blue", markersize=13,
                    markerfacecolor="none", markeredgewidth=2, zorder=6)
            ax.plot([snap["q_near"][0], snap["q_rand"][0]],
                    [snap["q_near"][1], snap["q_rand"][1]], "--", color="blue", linewidth=1.5, zorder=3)
        elif t == "steer":
            ax.plot(*snap["q_rand"], "x", color="darkorange", markersize=14, mew=3, zorder=6, alpha=0.5)
            ax.plot([snap["q_near"][0], snap["q_rand"][0]],
                    [snap["q_near"][1], snap["q_rand"][1]], "--", color="gray", alpha=0.5, zorder=3)
            ax.annotate("", xy=snap["q_new"], xytext=snap["q_near"],
                        arrowprops=dict(arrowstyle="-|>", color="darkviolet", lw=2.5), zorder=6)
            ax.plot(*snap["q_new"], "o", color="darkviolet", markersize=9, zorder=6)
        elif t == "collision":
            for (p, inside) in snap.get("viz_points", []):
                ax.plot(*p, ".", color=("red" if inside else "gray"),
                        markersize=6 if inside else 4, alpha=0.9 if inside else 0.6, zorder=5)
            color = "green" if snap["ok"] else "red"
            style = "-" if snap["ok"] else "--"
            ax.plot([snap["q_near"][0], snap["q_new"][0]],
                    [snap["q_near"][1], snap["q_new"][1]], style, color=color, linewidth=2.5, zorder=6)
            ax.plot(*snap["q_new"], "o", color=color, markersize=9, zorder=6)
            if not snap["ok"]:
                ax.plot(*snap["q_new"], "x", color="darkred", markersize=16, mew=3, zorder=7)
        elif t == "goalcheck":
            circle = patches.Circle(GOAL, GOAL_THRESHOLD, fill=False, linestyle="--",
                                     color="purple", linewidth=1.5, zorder=3)
            ax.add_patch(circle)
        elif t == "connect":
            for (p, inside) in snap.get("viz_points", []):
                ax.plot(*p, ".", color=("red" if inside else "gray"),
                        markersize=6 if inside else 4, alpha=0.9 if inside else 0.6, zorder=5)
            ax.plot([snap["q_new"][0], snap["goal"][0]],
                    [snap["q_new"][1], snap["goal"][1]], "-", color="green", linewidth=2.5, zorder=6)
        elif t == "path":
            xs = [p[0] for p in snap["path"]]
            ys = [p[1] for p in snap["path"]]
            ax.plot(xs, ys, "-", color="magenta", linewidth=3.5, zorder=7)
            ax.plot(xs, ys, "o", color="magenta", markersize=7, zorder=7)

        ax.set_title(snap["title"])

        # pseudocode panel
        axc = self.ax_code
        axc.clear()
        axc.axis("off")
        axc.set_xlim(0, 1)
        axc.set_ylim(0, len(CODE_LINES) + 1)
        axc.set_title("Pseudocode")
        hl = snap["highlight"]
        y_hl = len(CODE_LINES) - hl
        axc.add_patch(patches.Rectangle((0, y_hl - 0.42), 1, 0.84,
                                         color="yellow", alpha=0.45, zorder=0))
        for i, line in enumerate(CODE_LINES):
            y = len(CODE_LINES) - i
            axc.text(0.02, y, line, fontsize=11, family="monospace", va="center", zorder=1)

        # function detail panel
        axd = self.ax_detail
        axd.clear()
        axd.axis("off")
        axd.set_xlim(0, 1)
        axd.set_ylim(0, 1)
        axd.set_title("Function Detail")
        axd.text(0.02, 0.97, snap.get("detail", ""), fontsize=9, family="monospace",
                  va="top", ha="left", transform=axd.transAxes)

        self.status_text.set_text(
            f"Step {self.cursor + 1}/{len(self.history)}   |   {snap['info']}"
        )
        self.status_text.set_color(snap.get("info_color", "black"))

        back_state = "normal" if self.cursor > 0 else "disabled"
        self.btn_back.ax.set_facecolor("0.9" if back_state == "disabled" else "0.85")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive RRT step-through visualizer")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility (default: truly random each run)")
    parser.add_argument("--goal-bias", type=float, default=0.0,
                         help="Initial probability of sampling the goal directly each iteration "
                              "(default 0.0 = pure uniform sampling; toggle in the UI at any time)")
    args = parser.parse_args()

    app = App(seed=args.seed, goal_bias=args.goal_bias)
    app.show()


if __name__ == "__main__":
    main()
