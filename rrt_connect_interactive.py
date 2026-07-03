"""
Interactive RRT-Connect step-through visualizer.

Implements the exact published algorithm (Kuffner & LaValle, 2000,
"RRT-Connect: An Efficient Approach to Single-Query Path Planning"),
matching the pseudocode from the lecture slides:

    EXTEND(T, q)                      -- one cautious step toward q
    CONNECT(T, q)                     -- repeat EXTEND until not Advanced
    RRT_CONNECT_PLANNER(qinit, qgoal) -- alternate growing Ta / greedily
                                          connecting Tb, swapping each round

Pops up a matplotlib window with:
  - Left panel  : the world -- both trees (Tree-Start in blue, Tree-Goal
                  in green), the current sample/target, steer arrow, and
                  collision-check sample points, updated one micro-step
                  at a time.
  - Upper-right : all three pseudocode blocks (PLANNER / EXTEND / CONNECT)
                  stacked, with the line currently executing highlighted --
                  the highlight jumps between blocks as the call stack
                  moves (PLANNER calls EXTEND, EXTEND calls NEAREST/collision
                  checks, CONNECT repeatedly calls EXTEND, etc).
  - Lower-right : "Function Detail" -- the real arithmetic behind
                  NEAREST_NEIGHBOR / NEW_CONFIG (steer) / the collision
                  check / the Reached-Advanced-Trapped decision, plus the
                  Ta/Tb role bookkeeping around SWAP().
  - Buttons     : "<< Back" / "Next >>" / "Reset (new run)".
                  (Left/Right arrow keys and 'r' also work.)

RANDOM_CONFIG() is genuinely random (Python's `random` module, pure
uniform over the map -- the published algorithm does not bias sampling).
Pass --seed for a reproducible run.

Run:
    python rrt_connect_interactive.py
    python rrt_connect_interactive.py --seed 7
"""

import argparse
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

# ----------------------------------------------------------------------
# World setup (same world used throughout this series of examples)
# ----------------------------------------------------------------------
START = (1.0, 1.0)
GOAL = (9.0, 9.0)
BOUNDS = (0.0, 10.0)
STEP_SIZE = 2.0
OBSTACLE = dict(x=5.0, y=4.0, w=2.0, h=2.0)
MAX_K = 300              # "for k = 1 to K do" in the planner
MAX_CONNECT_STEPS = 300  # safety cap on CONNECT's repeat loop
COLLISION_SAMPLES = 200
COLLISION_VIZ_STRIDE = 5

TREE_START_COLOR = "steelblue"
TREE_GOAL_COLOR = "seagreen"
TARGET_COLOR = "darkorange"

# ----------------------------------------------------------------------
# Pseudocode, verbatim from the paper / lecture slides (BUILD_RRT is
# omitted -- the connect planner never calls it, it's shown in the
# source material only as background for plain RRT).
# ----------------------------------------------------------------------
PLANNER_LINES = [
    "Ta.init(qinit);  Tb.init(qgoal);",
    "for k = 1 to K do",
    "    qrand <- RANDOM_CONFIG();",
    "    if not (EXTEND(Ta,qrand) = Trapped) then",
    "        if (CONNECT(Tb,qnew) = Reached) then",
    "            Return PATH(Ta,Tb);",
    "    SWAP(Ta,Tb);",
    "Return Failure",
]
EXTEND_LINES = [
    "qnear <- NEAREST_NEIGHBOR(q,T);",
    "if NEW_CONFIG(q,qnear,qnew) then",
    "    T.add_vertex(qnew);",
    "    T.add_edge(qnear,qnew);",
    "    if qnew = q then",
    "        Return Reached;",
    "    else",
    "        Return Advanced;",
    "Return Trapped;",
]
CONNECT_LINES = [
    "repeat",
    "    S <- EXTEND(T,q);",
    "until not (S = Advanced)",
    "Return S;",
]
CODE_BLOCKS = [
    ("PLANNER", "RRT_CONNECT_PLANNER(qinit, qgoal):", PLANNER_LINES),
    ("EXTEND", "EXTEND(T, q):", EXTEND_LINES),
    ("CONNECT", "CONNECT(T, q):", CONNECT_LINES),
]


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def obstacle_contains(p):
    x, y = p
    return (OBSTACLE["x"] <= x <= OBSTACLE["x"] + OBSTACLE["w"] and
            OBSTACLE["y"] <= y <= OBSTACLE["y"] + OBSTACLE["h"])


def collision_check_detailed(p1, p2, samples=COLLISION_SAMPLES, viz_stride=COLLISION_VIZ_STRIDE):
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
            break
    return (first_collision is None), first_collision, viz_points


def nearest_node_detailed(nodes, q):
    dists = [dist(n["pos"], q) for n in nodes]
    near_i = min(range(len(nodes)), key=lambda i: dists[i])
    return near_i, dists[near_i], dists


def new_config_detailed(q_near, q, step):
    """NEW_CONFIG(q, qnear, qnew): steer at most `step` from q_near toward q.
    Returns (q_target, math_dict, reached) where reached=True means
    q_target == q exactly (within the step radius already)."""
    dx = q[0] - q_near[0]
    dy = q[1] - q_near[1]
    d = math.hypot(dx, dy)
    if d <= step:
        return q, dict(dx=dx, dy=dy, d=d, truncated=False), True
    ux, uy = dx / d, dy / d
    sx, sy = ux * step, uy * step
    q_target = (q_near[0] + sx, q_near[1] + sy)
    return q_target, dict(dx=dx, dy=dy, d=d, truncated=True, ux=ux, uy=uy, sx=sx, sy=sy), False


# ----------------------------------------------------------------------
# Tree
# ----------------------------------------------------------------------
class Tree:
    def __init__(self, root, name, color):
        self.nodes = [{"pos": root, "parent": None}]
        self.name = name
        self.color = color

    def add(self, pos, parent_i):
        self.nodes.append({"pos": pos, "parent": parent_i})
        return len(self.nodes) - 1

    def positions(self):
        return [n["pos"] for n in self.nodes]

    def edges(self):
        return [(self.nodes[n["parent"]]["pos"], n["pos"])
                for n in self.nodes if n["parent"] is not None]

    def path_to_root(self, node_i):
        path = []
        i = node_i
        while i is not None:
            path.append(self.nodes[i]["pos"])
            i = self.nodes[i]["parent"]
        path.reverse()
        return path


# ----------------------------------------------------------------------
# RRT-Connect engine, built as nested generators. Each generator yields
# UI snapshots and `return`s its algorithmic result -- `yield from` lets
# a parent capture that result while transparently forwarding all of the
# child's snapshots, so the flat history the UI sees still reflects the
# full call structure of PLANNER -> CONNECT -> EXTEND.
# ----------------------------------------------------------------------
class Engine:
    def __init__(self, rng):
        self.rng = rng
        self.tree_start = Tree(START, "Tree-Start", TREE_START_COLOR)
        self.tree_goal = Tree(GOAL, "Tree-Goal", TREE_GOAL_COLOR)
        self.result_path = None

    def _snap(self, Ta, Tb, **kw):
        snap = dict(
            ts_nodes=list(self.tree_start.positions()),
            ts_edges=list(self.tree_start.edges()),
            tg_nodes=list(self.tree_goal.positions()),
            tg_edges=list(self.tree_goal.edges()),
            Ta_name=Ta.name, Tb_name=Tb.name,
            title="", info="", info_color="black", detail="",
        )
        snap.update(kw)
        return snap

    # -- EXTEND(T, q) -----------------------------------------------------
    def extend(self, T, other, q, target_label, Ta, Tb):
        near_i, near_d, all_dists = nearest_node_detailed(T.nodes, q)
        q_near = T.nodes[near_i]["pos"]

        order = sorted(range(len(all_dists)), key=lambda i: all_dists[i])
        show_n = min(6, len(order))
        lines = [f"NEAREST_NEIGHBOR({target_label}, {T.name}): distance to each of {len(all_dists)} node(s)"]
        for idx in order[:show_n]:
            p = T.nodes[idx]["pos"]
            marker = "  <- NEAREST" if idx == near_i else ""
            lines.append(f"  node[{idx}] ({p[0]:.2f},{p[1]:.2f})  dist={all_dists[idx]:.2f}{marker}")
        if len(order) > show_n:
            lines.append(f"  ... and {len(order) - show_n} more node(s)")
        lines.append(f"  => qnear = node[{near_i}], dist={near_d:.2f}")
        yield self._snap(Ta, Tb, type="ext_nearest", highlight=("EXTEND", 1),
                          T_color=T.color, q=q, q_near=q_near, target_label=target_label,
                          title=f"EXTEND({T.name}, {target_label}): Nearest Neighbor",
                          info=f"qnear = ({q_near[0]:.2f}, {q_near[1]:.2f})",
                          info_color=T.color, detail="\n".join(lines))

        q_target, sd, reached_dist = new_config_detailed(q_near, q, STEP_SIZE)
        if sd["truncated"]:
            steer_detail = (
                f"NEW_CONFIG({target_label}, qnear, qnew) -- steer step={STEP_SIZE}:\n"
                f"  dx = {sd['dx']:.2f}, dy = {sd['dy']:.2f}, |d| = {sd['d']:.2f}  (> step)\n"
                f"  unit = ({sd['ux']:.3f}, {sd['uy']:.3f})\n"
                f"  step_vec = step * unit = ({sd['sx']:.3f}, {sd['sy']:.3f})\n"
                f"  qnew(candidate) = qnear + step_vec = ({q_target[0]:.2f}, {q_target[1]:.2f})"
            )
        else:
            steer_detail = (
                f"NEW_CONFIG({target_label}, qnear, qnew) -- steer step={STEP_SIZE}:\n"
                f"  |d| = {sd['d']:.2f}  (<= step, no truncation needed)\n"
                f"  qnew(candidate) = {target_label} = ({q_target[0]:.2f}, {q_target[1]:.2f})"
            )
        yield self._snap(Ta, Tb, type="ext_steer", highlight=("EXTEND", 2),
                          T_color=T.color, q=q, q_near=q_near, q_target=q_target, target_label=target_label,
                          title=f"EXTEND({T.name}, {target_label}): NEW_CONFIG (steer)",
                          info=f"qnew(candidate) = ({q_target[0]:.2f}, {q_target[1]:.2f})",
                          detail=steer_detail)

        ok, first_collision, viz_points = collision_check_detailed(q_near, q_target)
        ox0, oy0 = OBSTACLE["x"], OBSTACLE["y"]
        ox1, oy1 = ox0 + OBSTACLE["w"], oy0 + OBSTACLE["h"]
        if ok:
            coll_detail = (f"Collision check for segment qnear->qnew(candidate):\n"
                            f"  obstacle = x:[{ox0:.0f},{ox1:.0f}], y:[{oy0:.0f},{oy1:.0f}]\n"
                            f"  every sampled point is outside the obstacle\n"
                            f"  => NEW_CONFIG succeeds (True)")
        else:
            t, p = first_collision
            coll_detail = (f"Collision check for segment qnear->qnew(candidate):\n"
                            f"  obstacle = x:[{ox0:.0f},{ox1:.0f}], y:[{oy0:.0f},{oy1:.0f}]\n"
                            f"  first hit at t={t:.2f} -> ({p[0]:.2f},{p[1]:.2f}) is INSIDE the obstacle\n"
                            f"  => NEW_CONFIG fails (False)")
        yield self._snap(Ta, Tb, type="ext_collision", highlight=("EXTEND", 2),
                          T_color=T.color, q=q, q_near=q_near, q_target=q_target, ok=ok,
                          viz_points=viz_points, target_label=target_label,
                          title=f"EXTEND({T.name}, {target_label}): Collision Check",
                          info=("NEW_CONFIG = True" if ok else "NEW_CONFIG = False (Trapped)"),
                          info_color=("green" if ok else "red"), detail=coll_detail)

        if not ok:
            yield self._snap(Ta, Tb, type="ext_status", highlight=("EXTEND", 9),
                              T_color=T.color, status="Trapped",
                              title=f"EXTEND({T.name}, {target_label}) -> Trapped",
                              info="Trapped: collision, no vertex added.",
                              info_color="red",
                              detail="NEW_CONFIG failed -> no vertex/edge added.\nReturn Trapped;")
            return ("Trapped", None, None)

        new_i = T.add(q_target, near_i)
        yield self._snap(Ta, Tb, type="ext_add", highlight=("EXTEND", [3, 4]),
                          T_color=T.color, q_near=q_near, q_new=q_target,
                          title=f"EXTEND({T.name}, {target_label}): Add Vertex + Edge",
                          info=f"{T.name}.add_vertex({q_target[0]:.2f},{q_target[1]:.2f}); add_edge(qnear,qnew)",
                          info_color=T.color,
                          detail=(f"T.add_vertex(qnew);  T.add_edge(qnear,qnew);\n"
                                  f"  new node[{new_i}] = ({q_target[0]:.2f}, {q_target[1]:.2f}) in {T.name}\n"
                                  f"  parent = node[{near_i}]\n"
                                  f"  {T.name} size now = {len(T.nodes)}"))

        if reached_dist:
            yield self._snap(Ta, Tb, type="ext_status", highlight=("EXTEND", 6),
                              T_color=T.color, status="Reached", q_new=q_target,
                              title=f"EXTEND({T.name}, {target_label}) -> Reached",
                              info=f"qnew = {target_label} exactly -> Reached",
                              info_color="darkgreen",
                              detail=f"qnew = {target_label}?  YES\nReturn Reached;")
            return ("Reached", q_target, new_i)
        else:
            yield self._snap(Ta, Tb, type="ext_status", highlight=("EXTEND", 8),
                              T_color=T.color, status="Advanced", q_new=q_target,
                              title=f"EXTEND({T.name}, {target_label}) -> Advanced",
                              info=f"qnew != {target_label} -> Advanced",
                              info_color="darkorange",
                              detail=f"qnew = {target_label}?  NO (moved only {STEP_SIZE} units)\nReturn Advanced;")
            return ("Advanced", q_target, new_i)

    # -- CONNECT(T, q) ------------------------------------------------------
    def connect(self, T, other, q, target_label, Ta, Tb):
        yield self._snap(Ta, Tb, type="conn_repeat", highlight=("CONNECT", 1),
                          T_color=T.color,
                          title=f"CONNECT({T.name}, {target_label}): repeat",
                          info="repeat: keep calling EXTEND until it doesn't just Advance",
                          detail="repeat\n  S <- EXTEND(T,q);\nuntil not (S = Advanced)")

        steps = 0
        status, q_pos, node_i = "Advanced", None, None
        while status == "Advanced":
            steps += 1
            if steps > MAX_CONNECT_STEPS:
                status = "Trapped"
                break
            status, q_pos, node_i = yield from self.extend(T, other, q, target_label, Ta, Tb)
            yield self._snap(Ta, Tb, type="conn_result", highlight=("CONNECT", 2),
                              T_color=T.color, status=status,
                              title=f"CONNECT({T.name}, {target_label}): S = EXTEND(...)",
                              info=f"S = {status}",
                              info_color={"Reached": "darkgreen", "Advanced": "darkorange", "Trapped": "red"}[status],
                              detail=f"S <- EXTEND(T,q);\n  S = {status}")
            cont = (status == "Advanced")
            yield self._snap(Ta, Tb, type="conn_until", highlight=("CONNECT", 3),
                              T_color=T.color, status=status,
                              title=f"CONNECT({T.name}, {target_label}): until not (S=Advanced)",
                              info=("S = Advanced -> loop again" if cont else f"S = {status} -> exit loop"),
                              detail=(f"until not (S = Advanced):\n  S = {status}\n  "
                                      + ("condition FALSE -> repeat" if cont else "condition TRUE -> stop")))

        yield self._snap(Ta, Tb, type="conn_return", highlight=("CONNECT", 4),
                          T_color=T.color, status=status,
                          title=f"CONNECT({T.name}, {target_label}) -> {status}",
                          info=f"Return S = {status}",
                          info_color={"Reached": "darkgreen", "Advanced": "darkorange", "Trapped": "red"}[status],
                          detail=f"Return S;\n  S = {status}")
        return status, q_pos, node_i

    # -- RRT_CONNECT_PLANNER(qinit, qgoal) --------------------------------
    def run(self):
        Ta, Tb = self.tree_start, self.tree_goal
        yield self._snap(Ta, Tb, type="plan_init", highlight=("PLANNER", 1),
                          title="Initialize both trees",
                          info=f"Ta({Ta.name}) = {{{START}}},  Tb({Tb.name}) = {{{GOAL}}}",
                          detail=f"Ta.init(qinit);  Tb.init(qgoal);\n  qinit = {START}\n  qgoal = {GOAL}")

        k = 0
        while k < MAX_K:
            k += 1
            yield self._snap(Ta, Tb, type="plan_loop", highlight=("PLANNER", 2),
                              title=f"Iteration k={k}",
                              info=f"Ta = {Ta.name} ({len(Ta.nodes)} nodes),  Tb = {Tb.name} ({len(Tb.nodes)} nodes)",
                              detail=f"for k = 1 to K do   (k={k})\n  Ta = {Ta.name}, Tb = {Tb.name}")

            q_rand = (self.rng.uniform(*BOUNDS), self.rng.uniform(*BOUNDS))
            yield self._snap(Ta, Tb, type="plan_sample", highlight=("PLANNER", 3),
                              q_rand=q_rand,
                              title=f"Iteration {k}: RANDOM_CONFIG()",
                              info=f"qrand = ({q_rand[0]:.2f}, {q_rand[1]:.2f})",
                              info_color=TARGET_COLOR,
                              detail=(f"qrand <- RANDOM_CONFIG():\n"
                                      f"  x ~ Uniform{BOUNDS} = {q_rand[0]:.3f}\n"
                                      f"  y ~ Uniform{BOUNDS} = {q_rand[1]:.3f}"))

            yield self._snap(Ta, Tb, type="plan_call_extend", highlight=("PLANNER", 4),
                              q_rand=q_rand,
                              title=f"Iteration {k}: call EXTEND(Ta, qrand)",
                              info=f"Calling EXTEND({Ta.name}, qrand) ...",
                              detail=f"if not (EXTEND(Ta,qrand) = Trapped) then\n  entering EXTEND({Ta.name}, qrand)")

            status_a, q_new, new_i_a = yield from self.extend(Ta, Tb, q_rand, "qrand", Ta, Tb)

            not_trapped = status_a != "Trapped"
            yield self._snap(Ta, Tb, type="plan_extend_result", highlight=("PLANNER", 4),
                              q_rand=q_rand,
                              title=f"Iteration {k}: EXTEND(Ta,qrand) returned {status_a}",
                              info=f"not (EXTEND=Trapped)  =  {not_trapped}",
                              info_color=("darkgreen" if not_trapped else "red"),
                              detail=(f"if not (EXTEND(Ta,qrand) = Trapped) then\n"
                                      f"  EXTEND returned {status_a}\n"
                                      f"  condition = {not_trapped}"))

            if not_trapped:
                yield self._snap(Ta, Tb, type="plan_call_connect", highlight=("PLANNER", 5),
                                  q_new=q_new,
                                  title=f"Iteration {k}: call CONNECT(Tb, qnew)",
                                  info=f"Calling CONNECT({Tb.name}, qnew) ...",
                                  detail=f"if (CONNECT(Tb,qnew) = Reached) then\n  entering CONNECT({Tb.name}, qnew)")

                status_b, q_reach, new_i_b = yield from self.connect(Tb, Ta, q_new, "qnew", Ta, Tb)

                is_reached = status_b == "Reached"
                yield self._snap(Ta, Tb, type="plan_connect_result", highlight=("PLANNER", 5),
                                  q_new=q_new,
                                  title=f"Iteration {k}: CONNECT(Tb,qnew) returned {status_b}",
                                  info=f"CONNECT(Tb,qnew) = Reached  =  {is_reached}",
                                  info_color=("darkgreen" if is_reached else "black"),
                                  detail=(f"if (CONNECT(Tb,qnew) = Reached) then\n"
                                          f"  CONNECT returned {status_b}\n"
                                          f"  condition = {is_reached}"))

                if is_reached:
                    path_a = Ta.path_to_root(new_i_a)                    # Ta.root -> qnew
                    path_b = Tb.path_to_root(new_i_b)                    # Tb.root -> qnew
                    combined = path_a + list(reversed(path_b))[1:]       # Ta.root -> qnew -> ... -> Tb.root
                    full_path = combined if Ta.nodes[0]["pos"] == START else list(reversed(combined))
                    self.result_path = full_path
                    yield self._snap(Ta, Tb, type="plan_success", highlight=("PLANNER", 6),
                                      path=full_path,
                                      title="Trees Connected -- PATH(Ta, Tb)",
                                      info=f"Meeting point = ({q_new[0]:.2f}, {q_new[1]:.2f})",
                                      info_color="magenta",
                                      detail=("Return PATH(Ta, Tb);\n"
                                              f"  stitched at meeting point ({q_new[0]:.2f},{q_new[1]:.2f})\n"
                                              f"  path length = {len(full_path)} node(s)"))
                    return full_path

            yield self._snap(Ta, Tb, type="plan_swap", highlight=("PLANNER", 7),
                              title=f"Iteration {k}: SWAP(Ta, Tb)",
                              info=f"Roles swap: Ta becomes {Tb.name}, Tb becomes {Ta.name}",
                              detail=f"SWAP(Ta,Tb):\n  before: Ta={Ta.name}, Tb={Tb.name}\n  after:  Ta={Tb.name}, Tb={Ta.name}")
            Ta, Tb = Tb, Ta

        yield self._snap(Ta, Tb, type="plan_failure", highlight=("PLANNER", 8),
                          title="Failure",
                          info=f"K={MAX_K} iterations exhausted without connecting the trees.",
                          info_color="red",
                          detail=f"Return Failure;\n  (K={MAX_K} exhausted)")
        return None


# ----------------------------------------------------------------------
# Pseudocode panel layout: flatten the 3 blocks into rows, remembering
# which (block, line) each row corresponds to so highlighting is a
# simple lookup.
# ----------------------------------------------------------------------
def build_code_rows():
    rows = []  # each row: (kind, text, block, line_no)
    for block, header, lines in CODE_BLOCKS:
        rows.append(("header", header, None, None))
        for i, line in enumerate(lines, start=1):
            rows.append(("code", line, block, i))
        rows.append(("blank", "", None, None))
    return rows


CODE_ROWS = build_code_rows()


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
class App:
    def __init__(self, seed):
        self.seed = seed
        self.history = []
        self.cursor = -1
        self._new_run()

        self.fig = plt.figure(figsize=(15.5, 9.5))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.9, 1.2], height_ratios=[1.7, 1.0],
                                    left=0.045, right=0.98, top=0.94, bottom=0.14,
                                    wspace=0.14, hspace=0.22)
        self.ax_world = self.fig.add_subplot(gs[:, 0])
        self.ax_code = self.fig.add_subplot(gs[0, 1])
        self.ax_detail = self.fig.add_subplot(gs[1, 1])

        ax_back = self.fig.add_axes([0.20, 0.03, 0.15, 0.055])
        ax_next = self.fig.add_axes([0.37, 0.03, 0.15, 0.055])
        ax_reset = self.fig.add_axes([0.54, 0.03, 0.22, 0.055])
        self.btn_back = Button(ax_back, "<< Back")
        self.btn_next = Button(ax_next, "Next >>")
        self.btn_reset = Button(ax_reset, "Reset (new run)")
        self.btn_back.on_clicked(self.on_back)
        self.btn_next.on_clicked(self.on_next)
        self.btn_reset.on_clicked(self.on_reset)

        self.status_text = self.fig.text(0.045, 0.09, "", fontsize=9, family="monospace")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.on_next(None)

    def _new_run(self):
        actual_seed = self.seed if self.seed is not None else random.SystemRandom().randrange(2**31)
        print(f"Starting new RRT-Connect run with seed={actual_seed}")
        self.engine = Engine(random.Random(actual_seed))
        self.gen = self.engine.run()
        self.history = []
        self.cursor = -1

    # -- navigation -------------------------------------------------------
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

    # -- drawing ------------------------------------------------------------
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
        ax.plot(*START, marker="*", color=TREE_START_COLOR, markersize=20,
                markeredgecolor="black", zorder=5)
        ax.annotate("Start (Tree-Start root)", START, textcoords="offset points", xytext=(8, -16))
        ax.plot(*GOAL, marker="*", color=TREE_GOAL_COLOR, markersize=20,
                markeredgecolor="black", zorder=5)
        ax.annotate("Goal (Tree-Goal root)", GOAL, textcoords="offset points", xytext=(-90, 10))

    def render(self):
        snap = self.history[self.cursor]
        ax = self.ax_world
        ax.clear()
        self._draw_static_world()

        for (p1, p2) in snap["ts_edges"]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=TREE_START_COLOR, linewidth=2, zorder=3)
        for p in snap["ts_nodes"]:
            ax.plot(*p, "o", color=TREE_START_COLOR, markersize=6, zorder=4)
        for (p1, p2) in snap["tg_edges"]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=TREE_GOAL_COLOR, linewidth=2, zorder=3)
        for p in snap["tg_nodes"]:
            ax.plot(*p, "o", color=TREE_GOAL_COLOR, markersize=6, zorder=4)

        t = snap["type"]
        if t == "plan_sample":
            ax.plot(*snap["q_rand"], "x", color=TARGET_COLOR, markersize=15, mew=3, zorder=6)
        elif t in ("plan_call_extend", "plan_extend_result") and "q_rand" in snap:
            ax.plot(*snap["q_rand"], "x", color=TARGET_COLOR, markersize=15, mew=3, zorder=6, alpha=0.7)
        elif t in ("plan_call_connect", "plan_connect_result") and "q_new" in snap:
            ax.plot(*snap["q_new"], "P", color=TARGET_COLOR, markersize=13, mew=1, zorder=6)
        elif t == "ext_nearest":
            for p in (snap["ts_nodes"] if snap["T_color"] == TREE_START_COLOR else snap["tg_nodes"]):
                ax.plot([p[0], snap["q"][0]], [p[1], snap["q"][1]], "-", color="gray",
                        alpha=0.25, linewidth=1, zorder=2)
            ax.plot(*snap["q"], "x" if snap["target_label"] == "qrand" else "P",
                    color=TARGET_COLOR, markersize=14, mew=2, zorder=6)
            ax.plot(*snap["q_near"], "o", color=snap["T_color"], markersize=13,
                    markerfacecolor="none", markeredgewidth=2.5, zorder=6)
            ax.plot([snap["q_near"][0], snap["q"][0]], [snap["q_near"][1], snap["q"][1]],
                    "--", color=snap["T_color"], linewidth=1.5, zorder=3)
        elif t == "ext_steer":
            ax.plot(*snap["q"], "x" if snap["target_label"] == "qrand" else "P",
                    color=TARGET_COLOR, markersize=14, mew=2, zorder=6, alpha=0.6)
            ax.plot([snap["q_near"][0], snap["q"][0]], [snap["q_near"][1], snap["q"][1]],
                    "--", color="gray", alpha=0.5, zorder=3)
            ax.annotate("", xy=snap["q_target"], xytext=snap["q_near"],
                        arrowprops=dict(arrowstyle="-|>", color="darkviolet", lw=2.5), zorder=6)
            ax.plot(*snap["q_target"], "o", color="darkviolet", markersize=9, zorder=6)
        elif t == "ext_collision":
            for (p, inside) in snap.get("viz_points", []):
                ax.plot(*p, ".", color=("red" if inside else "gray"),
                        markersize=6 if inside else 4, alpha=0.9 if inside else 0.6, zorder=5)
            color = "green" if snap["ok"] else "red"
            style = "-" if snap["ok"] else "--"
            ax.plot([snap["q_near"][0], snap["q_target"][0]], [snap["q_near"][1], snap["q_target"][1]],
                    style, color=color, linewidth=2.5, zorder=6)
            ax.plot(*snap["q_target"], "o", color=color, markersize=9, zorder=6)
            if not snap["ok"]:
                ax.plot(*snap["q_target"], "x", color="darkred", markersize=16, mew=3, zorder=7)
        elif t == "ext_add":
            ax.plot([snap["q_near"][0], snap["q_new"][0]], [snap["q_near"][1], snap["q_new"][1]],
                    "-", color=snap["T_color"], linewidth=2.5, zorder=6)
            ax.plot(*snap["q_new"], "o", color=snap["T_color"], markersize=9,
                    markeredgecolor="black", zorder=6)
        elif t == "plan_success":
            xs = [p[0] for p in snap["path"]]
            ys = [p[1] for p in snap["path"]]
            ax.plot(xs, ys, "-", color="magenta", linewidth=3.5, zorder=7)
            ax.plot(xs, ys, "o", color="magenta", markersize=7, zorder=7)

        ax.set_title(snap["title"])

        # pseudocode panel: 3 stacked blocks, highlight current (block,line)
        axc = self.ax_code
        axc.clear()
        axc.axis("off")
        n_rows = len(CODE_ROWS)
        axc.set_xlim(0, 1)
        axc.set_ylim(0, n_rows + 1)
        axc.set_title("Pseudocode  (PLANNER calls EXTEND / CONNECT)")

        hl_block, hl_line = snap.get("highlight", (None, None))
        hl_lines = hl_line if isinstance(hl_line, list) else ([hl_line] if hl_line is not None else [])

        for row_idx, (kind, text, block, line_no) in enumerate(CODE_ROWS):
            y = n_rows - row_idx
            if kind == "header":
                axc.text(0.0, y, text, fontsize=9.5, family="monospace", va="center",
                          fontweight="bold", zorder=1)
            elif kind == "code":
                if block == hl_block and line_no in hl_lines:
                    axc.add_patch(patches.Rectangle((0, y - 0.42), 1, 0.84,
                                                      color="yellow", alpha=0.5, zorder=0))
                axc.text(0.03, y, text, fontsize=8.7, family="monospace", va="center", zorder=1)

        # function detail panel
        axd = self.ax_detail
        axd.clear()
        axd.axis("off")
        axd.set_xlim(0, 1)
        axd.set_ylim(0, 1)
        axd.set_title("Function Detail")
        axd.text(0.02, 0.97, snap.get("detail", ""), fontsize=8.8, family="monospace",
                  va="top", ha="left", transform=axd.transAxes)

        role_line = f"Ta = {snap.get('Ta_name', '?')}   |   Tb = {snap.get('Tb_name', '?')}"
        self.status_text.set_text(
            f"Step {self.cursor + 1}/{len(self.history)}   |   {role_line}   |   {snap['info']}"
        )
        self.status_text.set_color(snap.get("info_color", "black"))

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive RRT-Connect step-through visualizer")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility (default: truly random each run)")
    args = parser.parse_args()
    App(seed=args.seed).show()


if __name__ == "__main__":
    main()
