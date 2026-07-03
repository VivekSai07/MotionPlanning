"""
RRT (Rapidly-exploring Random Tree) visualization.

This reproduces the exact worked example from the walkthrough: same q_rand
samples, same computed q_near / q_new values, the same rejected branch
(collision), and the same final path to the goal. Nothing here is randomly
generated -- every number below is taken directly from the walkthrough.

Produces:
  - rrt_animation.gif   : step-by-step animation of the algorithm running
  - rrt_final_tree.png  : static summary of the final tree + extracted path

Run:
    python rrt_visualization.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------------------------------------------------------------
# World setup (matches the problem statement)
# ----------------------------------------------------------------------
START = (1.0, 1.0)
GOAL = (9.0, 9.0)
STEP_SIZE = 2.0
GOAL_THRESHOLD = 1.0

# Obstacle rectangle (the "#######" block in the ASCII map, sitting
# roughly between rows y=4..6 and columns x=5..7).
OBSTACLE = dict(x=5.0, y=4.0, w=2.0, h=2.0)  # (x, y) = lower-left corner


def obstacle_contains(p, margin=0.0):
    x, y = p
    return (OBSTACLE["x"] - margin <= x <= OBSTACLE["x"] + OBSTACLE["w"] + margin and
            OBSTACLE["y"] - margin <= y <= OBSTACLE["y"] + OBSTACLE["h"] + margin)


def collision_free(p1, p2, samples=200):
    """Sample the segment p1->p2 and check every sample against the obstacle."""
    p1, p2 = np.array(p1), np.array(p2)
    for t in np.linspace(0.0, 1.0, samples):
        if obstacle_contains(p1 + t * (p2 - p1)):
            return False
    return True


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_node(nodes, q_rand):
    """Real nearest-neighbor search over the current tree."""
    best_i, best_d = 0, float("inf")
    for i, n in enumerate(nodes):
        d = dist(n["pos"], q_rand)
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


# ----------------------------------------------------------------------
# Node bookkeeping
# ----------------------------------------------------------------------
# Each node: {"pos": (x, y), "parent": index_or_None}
nodes = [{"pos": START, "parent": None}]

# The exact q_rand samples used in the walkthrough for iterations 1-4,
# and the exact q_new results that came out of Steer() for each.
# (near is *recomputed* here via real nearest-neighbor search -- it
# independently reproduces the same nearest node reported in the walkthrough.)
detailed_iterations = [
    {"q_rand": (8.0, 4.0), "q_new": (2.84, 1.78)},
    {"q_rand": (9.0, 8.0), "q_new": (4.25, 3.20)},
    {"q_rand": (8.0, 6.0), "q_new": (5.85, 4.39)},   # this one collides
    {"q_rand": (3.0, 8.0), "q_new": (3.5, 5.1)},
]

# After "many iterations" the walkthrough jumps straight to these nodes
# (no q_rand given for them in the source material -- they are added
# exactly as stated in the Path Extraction table).
fastforward_nodes = [
    (5.2, 6.8),
    (7.4, 8.0),
    (8.7, 8.6),
]

# ----------------------------------------------------------------------
# Run the (deterministic, hardcoded) algorithm and record every frame
# needed for the animation.
# ----------------------------------------------------------------------
# A "frame" is a dict describing what to draw. type is one of:
#   sample   -> show q_rand and a line from q_near to q_rand
#   steer    -> show the steered q_new and whether the edge collides
#   add      -> commit q_new as a permanent tree node
#   reject   -> mark q_new as rejected (collision), fades out
#   goalcheck-> show the goal-threshold circle test
#   connect  -> final edge into the goal
#   path     -> highlight the extracted path
frames = []

for it_num, it in enumerate(detailed_iterations, start=1):
    q_rand = it["q_rand"]
    near_i, near_d = nearest_node(nodes, q_rand)
    q_near = nodes[near_i]["pos"]
    q_new = it["q_new"]
    ok = collision_free(q_near, q_new)

    frames.append(dict(type="sample", it=it_num, q_rand=q_rand,
                        q_near=q_near, near_i=near_i, near_d=near_d))
    frames.append(dict(type="steer", it=it_num, q_near=q_near,
                        q_new=q_new, ok=ok))
    if ok:
        nodes.append({"pos": q_new, "parent": near_i})
        frames.append(dict(type="add", it=it_num, q_new=q_new,
                            parent=q_near, node_i=len(nodes) - 1))
    else:
        frames.append(dict(type="reject", it=it_num, q_new=q_new,
                            parent=q_near))

# Fast-forward through the remaining path nodes (as given directly by
# the walkthrough's Path Extraction table).
last_i = len(nodes) - 1
for pos in fastforward_nodes:
    parent_pos = nodes[last_i]["pos"]
    ok = collision_free(parent_pos, pos)
    assert ok, f"unexpected collision on fast-forward edge to {pos}"
    nodes.append({"pos": pos, "parent": last_i})
    last_i = len(nodes) - 1
    frames.append(dict(type="add", it="ff", q_new=pos, parent=parent_pos,
                        node_i=last_i))

# Goal check on the last node added, (8.7, 8.6)
last_pos = nodes[last_i]["pos"]
d_goal = dist(last_pos, GOAL)
frames.append(dict(type="goalcheck", q_new=last_pos, d=d_goal))

# Final connection to the goal
ok = collision_free(last_pos, GOAL)
assert ok, "unexpected collision connecting to goal"
nodes.append({"pos": GOAL, "parent": last_i})
goal_i = len(nodes) - 1
frames.append(dict(type="connect", q_new=last_pos, goal=GOAL))

# Extract the path by walking parent pointers backward
path = []
i = goal_i
while i is not None:
    path.append(nodes[i]["pos"])
    i = nodes[i]["parent"]
path.reverse()
frames.append(dict(type="path", path=path))

print("Extracted path (start -> goal):")
for p in path:
    print(f"  ({p[0]:.2f}, {p[1]:.2f})")

# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))


def draw_static_world(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 11))
    ax.grid(True, linestyle=":", alpha=0.4)
    rect = patches.Rectangle((OBSTACLE["x"], OBSTACLE["y"]), OBSTACLE["w"], OBSTACLE["h"],
                              facecolor="0.25", edgecolor="black", zorder=2)
    ax.add_patch(rect)
    ax.plot(*START, marker="*", color="green", markersize=18, zorder=5)
    ax.annotate("Start (1,1)", START, textcoords="offset points", xytext=(8, -14))
    ax.plot(*GOAL, marker="*", color="red", markersize=18, zorder=5)
    ax.annotate("Goal (9,9)", GOAL, textcoords="offset points", xytext=(-55, 8))


# Permanent tree state, rebuilt incrementally as the animation advances
tree_nodes = [START]
tree_edges = []  # list of (p1, p2)


def render(frame):
    ax.clear()
    draw_static_world(ax)

    # always draw the committed tree so far
    for (p1, p2) in tree_edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="steelblue", linewidth=2, zorder=3)
    for p in tree_nodes:
        ax.plot(*p, "o", color="steelblue", markersize=6, zorder=4)

    t = frame["type"]

    if t == "sample":
        ax.set_title(f"Iteration {frame['it']}: Random Sample + Nearest Node")
        ax.plot(*frame["q_rand"], "x", color="darkorange", markersize=14, mew=3, zorder=6)
        ax.annotate(f"q_rand={frame['q_rand']}", frame["q_rand"],
                    textcoords="offset points", xytext=(8, 6), color="darkorange")
        ax.plot(*frame["q_near"], "o", color="blue", markersize=12,
                markerfacecolor="none", markeredgewidth=2, zorder=6)
        ax.plot([frame["q_near"][0], frame["q_rand"][0]],
                [frame["q_near"][1], frame["q_rand"][1]],
                "--", color="gray", linewidth=1, zorder=3)
        ax.text(0.02, 0.98, f"q_near = {frame['q_near']}  (dist={frame['near_d']:.2f})",
                transform=ax.transAxes, va="top", fontsize=9)

    elif t == "steer":
        ax.set_title(f"Iteration {frame['it']}: Steer (Δq = {STEP_SIZE}) + Collision Check")
        color = "green" if frame["ok"] else "red"
        style = "-" if frame["ok"] else "--"
        ax.plot([frame["q_near"][0], frame["q_new"][0]],
                [frame["q_near"][1], frame["q_new"][1]],
                style, color=color, linewidth=2.5, zorder=6)
        ax.plot(*frame["q_new"], "o", color=color, markersize=9, zorder=6)
        label = "COLLISION" if not frame["ok"] else "collision-free"
        ax.text(0.02, 0.98, f"q_new = {frame['q_new']}  -> {label}",
                transform=ax.transAxes, va="top", fontsize=9, color=color)
        if not frame["ok"]:
            ax.plot(*frame["q_new"], "x", color="darkred", markersize=16, mew=3, zorder=7)

    elif t == "add":
        label = f"Iteration {frame['it']}: Add Node" if frame["it"] != "ff" else "Extending tree toward goal"
        ax.set_title(label)
        tree_edges.append((frame["parent"], frame["q_new"]))
        tree_nodes.append(frame["q_new"])
        for (p1, p2) in tree_edges[-1:]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="steelblue", linewidth=2, zorder=3)
        ax.plot(*frame["q_new"], "o", color="steelblue", markersize=8, zorder=6)
        ax.text(0.02, 0.98, f"Added {frame['q_new']} to tree", transform=ax.transAxes,
                va="top", fontsize=9, color="darkgreen")

    elif t == "reject":
        ax.set_title(f"Iteration {frame['it']}: Node Rejected (Collision)")
        ax.plot([frame["parent"][0], frame["q_new"][0]],
                [frame["parent"][1], frame["q_new"][1]],
                "--", color="red", alpha=0.5, linewidth=2, zorder=3)
        ax.plot(*frame["q_new"], "x", color="darkred", markersize=14, mew=3, zorder=6)
        ax.text(0.02, 0.98, f"Discarded {frame['q_new']} (crosses obstacle)",
                transform=ax.transAxes, va="top", fontsize=9, color="darkred")

    elif t == "goalcheck":
        ax.set_title("Goal Check")
        circle = patches.Circle(GOAL, GOAL_THRESHOLD, fill=False, linestyle="--",
                                 color="purple", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.plot(*frame["q_new"], "o", color="steelblue", markersize=9, zorder=6)
        ax.text(0.02, 0.98,
                f"dist({frame['q_new']}, Goal) = {frame['d']:.2f} < threshold={GOAL_THRESHOLD} -> try connect",
                transform=ax.transAxes, va="top", fontsize=9, color="purple")

    elif t == "connect":
        ax.set_title("Final Connection to Goal")
        ax.plot([frame["q_new"][0], frame["goal"][0]],
                [frame["q_new"][1], frame["goal"][1]],
                "-", color="green", linewidth=2.5, zorder=6)
        tree_edges.append((frame["q_new"], frame["goal"]))
        tree_nodes.append(frame["goal"])
        ax.text(0.02, 0.98, "Goal reached!", transform=ax.transAxes,
                va="top", fontsize=10, color="darkgreen", weight="bold")

    elif t == "path":
        ax.set_title("Final Extracted Path (Start -> Goal)")
        xs = [p[0] for p in frame["path"]]
        ys = [p[1] for p in frame["path"]]
        ax.plot(xs, ys, "-", color="magenta", linewidth=3.5, zorder=7)
        ax.plot(xs, ys, "o", color="magenta", markersize=7, zorder=7)

    return []


anim = FuncAnimation(fig, render, frames=frames, interval=1400, repeat=True)

gif_path = "rrt_animation.gif"
anim.save(gif_path, writer=PillowWriter(fps=0.9))
print(f"Saved animation -> {gif_path}")

# Final static summary figure
fig2, ax2 = plt.subplots(figsize=(7, 7))
ax = ax2
draw_static_world(ax)
for (p1, p2) in tree_edges:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="steelblue", linewidth=2, zorder=3)
for p in tree_nodes:
    ax.plot(*p, "o", color="steelblue", markersize=6, zorder=4)
xs = [p[0] for p in path]
ys = [p[1] for p in path]
ax.plot(xs, ys, "-", color="magenta", linewidth=3.5, zorder=7, label="Final path")
ax.plot(xs, ys, "o", color="magenta", markersize=7, zorder=7)
ax.legend(loc="lower right")
ax.set_title("RRT: Final Tree and Extracted Path")
png_path = "rrt_final_tree.png"
fig2.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"Saved final tree -> {png_path}")

plt.close(fig)
plt.close(fig2)
