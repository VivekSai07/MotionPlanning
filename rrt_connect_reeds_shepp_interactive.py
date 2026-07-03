"""
Interactive RRT-Connect + Reeds-Shepp steering visualizer.

This is the integration the previous two scripts were building toward:
RRT-Connect's EXTEND() now steers with a Reeds-Shepp curve instead of a
straight line, exactly as described in the course exercise --

    "the EXTEND step uses a kinematically feasible steering function
    (Reeds-Shepp motion) instead of Euclidean interpolation."

Concretely, compared to rrt_connect_interactive.py:
  - A tree node is now a full SE(2) pose (x, y, theta), not just (x, y).
  - NEAREST_NEIGHBOR's distance metric is the length of the shortest
    Reeds-Shepp path to the target -- exactly what
    ompl::base::ReedsSheppStateSpace::distance() does.
  - NEW_CONFIG (the "Steer" step) computes the full Reeds-Shepp path from
    qnear toward the target (reusing generate_candidates() from
    reeds_shepp_interactive.py), then truncates it to at most `range`
    arc-length -- mirroring OMPL's actual RRTConnect::growTree(), which
    interpolates along the state-space's path at fraction
    maxDistance / distance when the sample is farther than maxDistance.
  - CollisionFree() samples along that (possibly truncated) Reeds-Shepp
    curve, segment by segment, instead of along a line.
  - The final PATH(Ta, Tb) is a real driveable curve (arcs + straight
    segments, forward and backward), not a polyline.

The RRT-Connect tree/EXTEND/CONNECT/SWAP control structure and pseudocode
panel are reused from rrt_connect_interactive.py; the Reeds-Shepp
path-synthesis math is reused from reeds_shepp_interactive.py. Both are
imported as modules rather than re-derived, so there is exactly one
implementation of each piece of math in this repo.

Pops up a matplotlib window with the same layout as the previous two
scripts:
  - Left panel  : both trees (Tree-Start blue, Tree-Goal green) as actual
                  driven curves, turning circles, collision-check samples.
  - Upper-right : the PLANNER / EXTEND / CONNECT pseudocode, highlighted.
  - Lower-right : Function Detail -- the RS-distance table for
                  NEAREST_NEIGHBOR, the RS candidate table + truncation
                  math for NEW_CONFIG, and the segment-by-segment
                  collision check.
  - Buttons     : "<< Back" / "Next >>" / "Reset (new run)".

Run:
    python rrt_connect_reeds_shepp_interactive.py
    python rrt_connect_reeds_shepp_interactive.py --seed 3 --rho 2.0 --range 6.0
"""

import argparse
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

import reeds_shepp_interactive as rs
from rrt_connect_interactive import CODE_ROWS

# ----------------------------------------------------------------------
# World setup (same map as the rest of the series)
# ----------------------------------------------------------------------
BOUNDS = rs.BOUNDS
OBSTACLE = rs.OBSTACLE
START = (1.0, 1.0, 0.0)
GOAL = (9.0, 9.0, math.pi / 2)
DEFAULT_RHO = 2.0
DEFAULT_RANGE = 6.0          # max arc-length a single EXTEND may advance
MAX_K = 300
MAX_CONNECT_STEPS = 300
SAMPLES_PER_SEGMENT = 30
CANDIDATES_SHOWN = rs.CANDIDATES_SHOWN

TREE_START_COLOR = "steelblue"
TREE_GOAL_COLOR = "seagreen"
TARGET_COLOR = "darkorange"


# ----------------------------------------------------------------------
# Small SE(2) helpers built on top of reeds_shepp_interactive's core math
# ----------------------------------------------------------------------
def to_global(q0, lx, ly, lyaw):
    c, s = math.cos(q0[2]), math.sin(q0[2])
    gx = c * lx - s * ly + q0[0]
    gy = s * lx + c * ly + q0[1]
    gyaw = rs.pi_2_pi(lyaw + q0[2])
    return gx, gy, gyaw


def interpolate_along_candidate(candidate, rho, max_curvature, s_limit):
    """Walk candidate['ctypes']/['lengths'] (local frame relative to the
    RS candidate's q0) up to real arc-length s_limit. Returns the local
    pose reached and the list of (ctype, used_raw_len) segments actually
    driven -- the last one truncated if s_limit falls inside it."""
    ox, oy, oyaw = 0.0, 0.0, 0.0
    used = []
    remaining = s_limit
    for ctype, raw_len in zip(candidate["ctypes"], candidate["lengths"]):
        seg_real = abs(raw_len) * rho
        if seg_real <= remaining + 1e-9:
            used.append((ctype, raw_len))
            ox, oy, oyaw = rs.interpolate(raw_len, ctype, max_curvature, ox, oy, oyaw)
            remaining -= seg_real
            if remaining <= 1e-9:
                break
        else:
            frac = remaining / seg_real if seg_real > 1e-12 else 0.0
            partial_len = raw_len * frac
            used.append((ctype, partial_len))
            ox, oy, oyaw = rs.interpolate(partial_len, ctype, max_curvature, ox, oy, oyaw)
            break
    return (ox, oy, oyaw), used


def discretize_used_segments(q_near, used_segments, max_curvature, samples_per_segment=SAMPLES_PER_SEGMENT):
    """Discretize the (possibly truncated) segment list produced by
    interpolate_along_candidate into global-frame samples, collision
    checking as we go. Returns (ok, collision_point, seg_records) where
    seg_records is per-segment {ctype, center_global_or_None, samples}."""
    ox, oy, oyaw = 0.0, 0.0, 0.0
    seg_records = []
    collided_at = None
    for ctype, raw_len in used_segments:
        center_global = None
        if ctype in ("L", "R"):
            center_local = rs.turning_center(ox, oy, oyaw, 1.0 / max_curvature, ctype)
            center_global = to_global(q_near, center_local[0], center_local[1], 0.0)[:2]
        samples = []
        n = samples_per_segment
        seg_collided = False
        for i in range(n + 1):
            dist = raw_len * i / n if n > 0 else raw_len
            lx, ly, lyaw = rs.interpolate(dist, ctype, max_curvature, ox, oy, oyaw)
            gx, gy, gyaw = to_global(q_near, lx, ly, lyaw)
            inside = rs.obstacle_contains((gx, gy))
            samples.append((gx, gy, gyaw, inside))
            if inside:
                if collided_at is None:
                    collided_at = (gx, gy)
                seg_collided = True
                break
        seg_records.append(dict(ctype=ctype, raw_len=raw_len, center=center_global,
                                 samples=samples, forward=raw_len >= 0))
        if seg_collided:
            break
        ox, oy, oyaw = rs.interpolate(raw_len, ctype, max_curvature, ox, oy, oyaw)
    return collided_at is None, collided_at, seg_records


# ----------------------------------------------------------------------
# Tree (SE(2) poses; each non-root node also stores the discretized,
# already-collision-verified samples of the edge from its parent, so the
# final path can be redrawn as the true driven curve)
# ----------------------------------------------------------------------
class Tree:
    def __init__(self, root, name, color):
        self.nodes = [{"pos": root, "parent": None, "in_samples": []}]
        self.name = name
        self.color = color

    def add(self, pose, parent_i, in_samples):
        self.nodes.append({"pos": pose, "parent": parent_i, "in_samples": in_samples})
        return len(self.nodes) - 1

    def positions(self):
        return [n["pos"] for n in self.nodes]

    def all_edge_samples(self):
        out = []
        for n in self.nodes:
            if n["in_samples"]:
                out.append(n["in_samples"])
        return out

    def path_to_root(self, node_i):
        chain = []
        i = node_i
        while i is not None:
            chain.append(i)
            i = self.nodes[i]["parent"]
        chain.reverse()
        return chain


# ----------------------------------------------------------------------
# Engine: nested generators mirroring rrt_connect_interactive.py's
# structure, with EXTEND's internals replaced by Reeds-Shepp steering.
# ----------------------------------------------------------------------
class Engine:
    def __init__(self, rng, rho, range_limit):
        self.rng = rng
        self.rho = rho
        self.range_limit = range_limit
        self.max_curvature = 1.0 / rho
        self.tree_start = Tree(START, "Tree-Start", TREE_START_COLOR)
        self.tree_goal = Tree(GOAL, "Tree-Goal", TREE_GOAL_COLOR)
        self.result_path = None

    def _snap(self, Ta, Tb, **kw):
        snap = dict(
            ts_edges=self.tree_start.all_edge_samples(),
            tg_edges=self.tree_goal.all_edge_samples(),
            ts_nodes=[n["pos"] for n in self.tree_start.nodes],
            tg_nodes=[n["pos"] for n in self.tree_goal.nodes],
            Ta_name=Ta.name, Tb_name=Tb.name,
            title="", info="", info_color="black", detail="",
        )
        snap.update(kw)
        return snap

    # -- EXTEND(T, q): Reeds-Shepp steering ---------------------------------
    def extend(self, T, q_target, target_label, Ta, Tb):
        # NEAREST_NEIGHBOR under the Reeds-Shepp distance metric
        dists, cands_per_node = [], []
        for node in T.nodes:
            cands = rs.generate_candidates(node["pos"], q_target, self.max_curvature)
            d = cands[0]["total_len"] if cands else float("inf")
            dists.append(d)
            cands_per_node.append(cands)
        near_i = min(range(len(dists)), key=lambda i: dists[i])
        q_near = T.nodes[near_i]["pos"]
        near_d = dists[near_i]
        candidates = cands_per_node[near_i]

        order = sorted(range(len(dists)), key=lambda i: dists[i])
        show_n = min(6, len(order))
        lines = [f"NEAREST_NEIGHBOR({target_label}, {T.name}): RS-path length to each of {len(dists)} node(s)",
                 "(same metric as ompl::ReedsSheppStateSpace::distance())"]
        for idx in order[:show_n]:
            p = T.nodes[idx]["pos"]
            marker = "  <- NEAREST" if idx == near_i else ""
            lines.append(f"  node[{idx}] ({p[0]:.2f},{p[1]:.2f},{math.degrees(p[2]):.0f}°)  "
                         f"RSdist={dists[idx]:.2f}{marker}")
        if len(order) > show_n:
            lines.append(f"  ... and {len(order) - show_n} more node(s)")
        lines.append(f"  => qnear = node[{near_i}], RSdist={near_d:.2f}")
        yield self._snap(Ta, Tb, type="ext_nearest", highlight=("EXTEND", 1),
                          T_color=T.color, q_near=q_near, target_label=target_label,
                          title=f"EXTEND({T.name}, {target_label}): Nearest Neighbor (RS metric)",
                          info=f"qnear = ({q_near[0]:.2f},{q_near[1]:.2f},{math.degrees(q_near[2]):.0f}°)",
                          info_color=T.color, detail="\n".join(lines))

        # RS candidate enumeration for the winning (qnear, target) pair
        clines = [f"generate_candidates(qnear, {target_label}): {len(candidates)} valid RS word(s)"]
        for i, c in enumerate(candidates[:CANDIDATES_SHOWN]):
            label = " ".join(f"{t}{'+' if l >= 0 else '-'}" for t, l in zip(c["ctypes"], c["lengths"]))
            marker = "  <- SHORTEST" if i == 0 else ""
            clines.append(f"  [{label:<16s}] length={c['total_len']:.2f}{marker}")
        if len(candidates) > CANDIDATES_SHOWN:
            clines.append(f"  ... and {len(candidates) - CANDIDATES_SHOWN} more")
        yield self._snap(Ta, Tb, type="ext_candidates", highlight=("EXTEND", 2),
                          T_color=T.color, q_near=q_near, q_target=q_target, target_label=target_label,
                          candidates=candidates,
                          title=f"EXTEND({T.name}, {target_label}): RS Candidate Enumeration",
                          info=f"{len(candidates)} valid RS candidates for qnear -> {target_label}",
                          detail="\n".join(clines))

        winner = candidates[0]
        total_len = winner["total_len"]
        reached = total_len <= self.range_limit + 1e-9
        s_used = total_len if reached else self.range_limit
        (lx, ly, lyaw), used_segments = interpolate_along_candidate(winner, self.rho, self.max_curvature, s_used)
        q_new = to_global(q_near, lx, ly, lyaw)

        seg_lines = [f"NEW_CONFIG: winner length={total_len:.2f}, range={self.range_limit:.2f}"]
        if reached:
            seg_lines.append(f"  length <= range -> drive the FULL path -> qnew = {target_label}")
        else:
            seg_lines.append(f"  length > range -> truncate to s={s_used:.2f} (Advanced, not Reached)")
        seg_lines.append(f"  qnew(candidate) = ({q_new[0]:.2f}, {q_new[1]:.2f}, {math.degrees(q_new[2]):.0f}°)")
        seg_lines.append("  segments actually driven:")
        for ctype, l in used_segments:
            seg_lines.append(f"    {ctype} {'forward ' if l >= 0 else 'backward'}  "
                              f"len={abs(l) * self.rho:.2f}m")
        yield self._snap(Ta, Tb, type="ext_steer", highlight=("EXTEND", 2),
                          T_color=T.color, q_near=q_near, q_target=q_new, target_label=target_label,
                          used_segments=used_segments, reached=reached,
                          title=f"EXTEND({T.name}, {target_label}): NEW_CONFIG (RS steer + truncate)",
                          info=f"qnew(candidate) = ({q_new[0]:.2f},{q_new[1]:.2f},{math.degrees(q_new[2]):.0f}°)",
                          detail="\n".join(seg_lines))

        ok, collided_at, seg_records = discretize_used_segments(q_near, used_segments, self.max_curvature)
        ox0, oy0 = OBSTACLE["x"], OBSTACLE["y"]
        ox1, oy1 = ox0 + OBSTACLE["w"], oy0 + OBSTACLE["h"]
        n_samples = sum(len(sr["samples"]) for sr in seg_records)
        if ok:
            coll_detail = (f"Collision check across {len(seg_records)} segment(s), {n_samples} sample(s):\n"
                            f"  obstacle = x:[{ox0:.0f},{ox1:.0f}], y:[{oy0:.0f},{oy1:.0f}]\n"
                            f"  every sampled (x,y) is outside the obstacle\n"
                            f"  => NEW_CONFIG succeeds (True)")
        else:
            coll_detail = (f"Collision check across {len(seg_records)} segment(s):\n"
                            f"  obstacle = x:[{ox0:.0f},{ox1:.0f}], y:[{oy0:.0f},{oy1:.0f}]\n"
                            f"  hit at ({collided_at[0]:.2f}, {collided_at[1]:.2f})\n"
                            f"  => NEW_CONFIG fails (False)")
        yield self._snap(Ta, Tb, type="ext_collision", highlight=("EXTEND", 2),
                          T_color=T.color, q_near=q_near, seg_records=seg_records, ok=ok,
                          title=f"EXTEND({T.name}, {target_label}): Collision Check (along RS curve)",
                          info=("NEW_CONFIG = True" if ok else "NEW_CONFIG = False (Trapped)"),
                          info_color=("green" if ok else "red"), detail=coll_detail)

        if not ok:
            yield self._snap(Ta, Tb, type="ext_status", highlight=("EXTEND", 9),
                              T_color=T.color, status="Trapped",
                              title=f"EXTEND({T.name}, {target_label}) -> Trapped",
                              info="Trapped: collision along the RS curve, no vertex added.",
                              info_color="red",
                              detail="NEW_CONFIG failed -> no vertex/edge added.\nReturn Trapped;")
            return "Trapped", None, None

        edge_samples = [(gx, gy, gyaw, sr["forward"])
                         for sr in seg_records for (gx, gy, gyaw, _inside) in sr["samples"]]
        new_i = T.add(q_new, near_i, edge_samples)
        yield self._snap(Ta, Tb, type="ext_add", highlight=("EXTEND", [3, 4]),
                          T_color=T.color, q_near=q_near, q_new=q_new, seg_records=seg_records,
                          title=f"EXTEND({T.name}, {target_label}): Add Vertex + Edge",
                          info=f"{T.name}.add_vertex(...); add_edge(qnear,qnew)  [{len(used_segments)} RS segment(s)]",
                          info_color=T.color,
                          detail=(f"T.add_vertex(qnew);  T.add_edge(qnear,qnew);\n"
                                  f"  new node[{new_i}] = ({q_new[0]:.2f},{q_new[1]:.2f},{math.degrees(q_new[2]):.0f}°)\n"
                                  f"  parent = node[{near_i}]\n"
                                  f"  edge = {len(used_segments)} RS segment(s)\n"
                                  f"  {T.name} size now = {len(T.nodes)}"))

        if reached:
            yield self._snap(Ta, Tb, type="ext_status", highlight=("EXTEND", 6),
                              T_color=T.color, status="Reached", q_new=q_new,
                              title=f"EXTEND({T.name}, {target_label}) -> Reached",
                              info=f"qnew = {target_label} exactly -> Reached",
                              info_color="darkgreen",
                              detail=f"qnew = {target_label}?  YES\nReturn Reached;")
            return "Reached", q_new, new_i
        else:
            yield self._snap(Ta, Tb, type="ext_status", highlight=("EXTEND", 8),
                              T_color=T.color, status="Advanced", q_new=q_new,
                              title=f"EXTEND({T.name}, {target_label}) -> Advanced",
                              info=f"qnew != {target_label} (range-limited) -> Advanced",
                              info_color="darkorange",
                              detail=f"qnew = {target_label}?  NO (advanced only {self.range_limit:.1f}m)\nReturn Advanced;")
            return "Advanced", q_new, new_i

    # -- CONNECT(T, q) --------------------------------------------------
    def connect(self, T, q_target, target_label, Ta, Tb):
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
            status, q_pos, node_i = yield from self.extend(T, q_target, target_label, Ta, Tb)
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

    # -- RRT_CONNECT_PLANNER(qinit, qgoal) -------------------------------
    def run(self):
        Ta, Tb = self.tree_start, self.tree_goal
        yield self._snap(Ta, Tb, type="plan_init", highlight=("PLANNER", 1),
                          title="Initialize both trees",
                          info=f"Ta({Ta.name}) = {{start}},  Tb({Tb.name}) = {{goal}}, rho={self.rho}, range={self.range_limit}",
                          detail=(f"Ta.init(qinit);  Tb.init(qgoal);\n"
                                  f"  qinit = ({START[0]},{START[1]},{math.degrees(START[2]):.0f}°)\n"
                                  f"  qgoal = ({GOAL[0]},{GOAL[1]},{math.degrees(GOAL[2]):.0f}°)\n"
                                  f"  rho (turning radius) = {self.rho}\n"
                                  f"  range (max arc-length per EXTEND) = {self.range_limit}"))

        k = 0
        while k < MAX_K:
            k += 1
            yield self._snap(Ta, Tb, type="plan_loop", highlight=("PLANNER", 2),
                              title=f"Iteration k={k}",
                              info=f"Ta = {Ta.name} ({len(Ta.nodes)} nodes),  Tb = {Tb.name} ({len(Tb.nodes)} nodes)",
                              detail=f"for k = 1 to K do   (k={k})\n  Ta = {Ta.name}, Tb = {Tb.name}")

            q_rand = (self.rng.uniform(*BOUNDS), self.rng.uniform(*BOUNDS), self.rng.uniform(-math.pi, math.pi))
            yield self._snap(Ta, Tb, type="plan_sample", highlight=("PLANNER", 3),
                              q_rand=q_rand,
                              title=f"Iteration {k}: RANDOM_CONFIG()",
                              info=f"qrand = ({q_rand[0]:.2f},{q_rand[1]:.2f},{math.degrees(q_rand[2]):.0f}°)",
                              info_color=TARGET_COLOR,
                              detail=(f"qrand <- RANDOM_CONFIG():\n"
                                      f"  x ~ Uniform{BOUNDS} = {q_rand[0]:.3f}\n"
                                      f"  y ~ Uniform{BOUNDS} = {q_rand[1]:.3f}\n"
                                      f"  theta ~ Uniform(-pi,pi) = {q_rand[2]:.3f} rad"))

            yield self._snap(Ta, Tb, type="plan_call_extend", highlight=("PLANNER", 4),
                              q_rand=q_rand,
                              title=f"Iteration {k}: call EXTEND(Ta, qrand)",
                              info=f"Calling EXTEND({Ta.name}, qrand) ...",
                              detail=f"if not (EXTEND(Ta,qrand) = Trapped) then\n  entering EXTEND({Ta.name}, qrand)")

            status_a, q_new, new_i_a = yield from self.extend(Ta, q_rand, "qrand", Ta, Tb)

            not_trapped = status_a != "Trapped"
            yield self._snap(Ta, Tb, type="plan_extend_result", highlight=("PLANNER", 4),
                              q_rand=q_rand,
                              title=f"Iteration {k}: EXTEND(Ta,qrand) returned {status_a}",
                              info=f"not (EXTEND=Trapped)  =  {not_trapped}",
                              info_color=("darkgreen" if not_trapped else "red"),
                              detail=(f"if not (EXTEND(Ta,qrand) = Trapped) then\n"
                                      f"  EXTEND returned {status_a}\n  condition = {not_trapped}"))

            if not_trapped:
                yield self._snap(Ta, Tb, type="plan_call_connect", highlight=("PLANNER", 5),
                                  q_new=q_new,
                                  title=f"Iteration {k}: call CONNECT(Tb, qnew)",
                                  info=f"Calling CONNECT({Tb.name}, qnew) ...",
                                  detail=f"if (CONNECT(Tb,qnew) = Reached) then\n  entering CONNECT({Tb.name}, qnew)")

                status_b, q_reach, new_i_b = yield from self.connect(Tb, q_new, "qnew", Ta, Tb)

                is_reached = status_b == "Reached"
                yield self._snap(Ta, Tb, type="plan_connect_result", highlight=("PLANNER", 5),
                                  q_new=q_new,
                                  title=f"Iteration {k}: CONNECT(Tb,qnew) returned {status_b}",
                                  info=f"CONNECT(Tb,qnew) = Reached  =  {is_reached}",
                                  info_color=("darkgreen" if is_reached else "black"),
                                  detail=(f"if (CONNECT(Tb,qnew) = Reached) then\n"
                                          f"  CONNECT returned {status_b}\n  condition = {is_reached}"))

                if is_reached:
                    chain_a = Ta.path_to_root(new_i_a)
                    samples_a = []
                    for idx in chain_a[1:]:
                        samples_a.extend(Ta.nodes[idx]["in_samples"])
                    chain_b = Tb.path_to_root(new_i_b)
                    samples_b = []
                    for idx in chain_b[1:]:
                        samples_b.extend(Tb.nodes[idx]["in_samples"])
                    samples_b_rev = [(x, y, yaw, not fwd) for (x, y, yaw, fwd) in reversed(samples_b)]
                    combined = samples_a + samples_b_rev
                    if Ta.nodes[0]["pos"] == START:
                        full_path = combined
                    else:
                        full_path = [(x, y, yaw, not fwd) for (x, y, yaw, fwd) in reversed(combined)]
                    self.result_path = full_path
                    total_length = len(full_path)  # sample count, real length computed below
                    yield self._snap(Ta, Tb, type="plan_success", highlight=("PLANNER", 6),
                                      path=full_path,
                                      title="Trees Connected -- PATH(Ta, Tb)",
                                      info=f"Meeting point = ({q_new[0]:.2f},{q_new[1]:.2f},{math.degrees(q_new[2]):.0f}°)",
                                      info_color="magenta",
                                      detail=("Return PATH(Ta, Tb);\n"
                                              f"  stitched at meeting point ({q_new[0]:.2f},{q_new[1]:.2f})\n"
                                              f"  {total_length} discretized pose samples along the final drive"))
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
# UI
# ----------------------------------------------------------------------
class App:
    def __init__(self, seed, rho, range_limit):
        self.seed = seed
        self.rho = rho
        self.range_limit = range_limit
        self.history = []
        self.cursor = -1
        self.engine = None
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
        print(f"Starting new RRT-Connect+Reeds-Shepp run with seed={actual_seed}")
        self.engine = Engine(random.Random(actual_seed), self.rho, self.range_limit)
        self.gen = self.engine.run()
        self.history = []
        self.cursor = -1

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
        ax.plot(START[0], START[1], marker="*", color=TREE_START_COLOR, markersize=20,
                markeredgecolor="black", zorder=5)
        ax.annotate("Start (q_init)", START[:2], textcoords="offset points", xytext=(8, -16))
        ax.plot(GOAL[0], GOAL[1], marker="*", color=TREE_GOAL_COLOR, markersize=20,
                markeredgecolor="black", zorder=5)
        ax.annotate("Goal (q_goal)", GOAL[:2], textcoords="offset points", xytext=(-90, 10))

    def _draw_edge_samples(self, ax, edge_samples, color):
        fwd_run, bwd_run = [], []
        last_fwd = None
        for (x, y, yaw, fwd) in edge_samples:
            if last_fwd is not None and fwd != last_fwd:
                self._flush_run(ax, fwd_run, bwd_run, color)
                fwd_run, bwd_run = [], []
            (fwd_run if fwd else bwd_run).append((x, y))
            last_fwd = fwd
        self._flush_run(ax, fwd_run, bwd_run, color)

    @staticmethod
    def _flush_run(ax, fwd_run, bwd_run, color):
        if len(fwd_run) > 1:
            xs, ys = zip(*fwd_run)
            ax.plot(xs, ys, "-", color=color, linewidth=2.3, zorder=5)
        if len(bwd_run) > 1:
            xs, ys = zip(*bwd_run)
            ax.plot(xs, ys, "--", color=color, linewidth=2.3, zorder=5)

    def render(self):
        snap = self.history[self.cursor]
        ax = self.ax_world
        ax.clear()
        self._draw_static_world()

        for edge_samples in snap["ts_edges"]:
            self._draw_edge_samples(ax, edge_samples, TREE_START_COLOR)
        for p in snap["ts_nodes"]:
            ax.plot(p[0], p[1], "o", color=TREE_START_COLOR, markersize=6, zorder=4)
        for edge_samples in snap["tg_edges"]:
            self._draw_edge_samples(ax, edge_samples, TREE_GOAL_COLOR)
        for p in snap["tg_nodes"]:
            ax.plot(p[0], p[1], "o", color=TREE_GOAL_COLOR, markersize=6, zorder=4)

        t = snap["type"]
        if t == "plan_sample":
            self._draw_pose_arrow(ax, snap["q_rand"], TARGET_COLOR)
        elif t in ("plan_call_extend", "plan_extend_result") and "q_rand" in snap:
            self._draw_pose_arrow(ax, snap["q_rand"], TARGET_COLOR, alpha=0.6)
        elif t in ("plan_call_connect", "plan_connect_result") and "q_new" in snap:
            self._draw_pose_arrow(ax, snap["q_new"], TARGET_COLOR)
        elif t in ("ext_nearest", "ext_candidates"):
            self._draw_pose_arrow(ax, snap["q_near"], snap["T_color"])
            if snap.get("q_target") is not None:
                self._draw_pose_arrow(ax, snap["q_target"], TARGET_COLOR, alpha=0.6)
        elif t == "ext_steer":
            self._draw_pose_arrow(ax, snap["q_near"], snap["T_color"])
            self._draw_pose_arrow(ax, snap["q_target"], "darkviolet")
            self._draw_used_segments_preview(ax, snap["q_near"], snap["used_segments"], "darkviolet")
        elif t == "ext_collision":
            self._draw_pose_arrow(ax, snap["q_near"], snap["T_color"])
            for sr in snap["seg_records"]:
                if sr["center"] is not None:
                    circ = patches.Circle(sr["center"], self.engine.rho, fill=False, linestyle="--",
                                           color="darkviolet", alpha=0.5, linewidth=1.2, zorder=3)
                    ax.add_patch(circ)
                for (gx, gy, gyaw, inside) in sr["samples"]:
                    ax.plot(gx, gy, ".", color=("red" if inside else "gray"),
                            markersize=7 if inside else 4, alpha=0.9 if inside else 0.6, zorder=6)
        elif t == "ext_add":
            self._draw_pose_arrow(ax, snap["q_new"], snap["T_color"])
        elif t == "plan_success":
            xs = [p[0] for p in snap["path"]]
            ys = [p[1] for p in snap["path"]]
            ax.plot(xs, ys, "-", color="magenta", linewidth=3, zorder=7, alpha=0.9)

        ax.set_title(snap["title"])

        # pseudocode panel (reused rows from rrt_connect_interactive.py)
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
        axd.text(0.02, 0.97, snap.get("detail", ""), fontsize=8.4, family="monospace",
                  va="top", ha="left", transform=axd.transAxes)

        role_line = f"Ta = {snap.get('Ta_name', '?')}   |   Tb = {snap.get('Tb_name', '?')}"
        self.status_text.set_text(
            f"Step {self.cursor + 1}/{len(self.history)}   |   {role_line}   |   {snap['info']}"
        )
        self.status_text.set_color(snap.get("info_color", "black"))

        self.fig.canvas.draw_idle()

    def _draw_pose_arrow(self, ax, pose, color, alpha=1.0):
        x, y, yaw = pose
        ax.annotate("", xy=(x + 1.0 * math.cos(yaw), y + 1.0 * math.sin(yaw)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, alpha=alpha), zorder=6)
        ax.plot(x, y, "o", color=color, markersize=6, alpha=alpha, zorder=6)

    def _draw_used_segments_preview(self, ax, q_near, used_segments, color):
        ox, oy, oyaw = 0.0, 0.0, 0.0
        for ctype, raw_len in used_segments:
            pts = []
            n = 20
            for i in range(n + 1):
                dist = raw_len * i / n if n > 0 else raw_len
                lx, ly, lyaw = rs.interpolate(dist, ctype, self.engine.max_curvature, ox, oy, oyaw)
                pts.append(to_global(q_near, lx, ly, lyaw)[:2])
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", color=color, alpha=0.5, linewidth=1.8, zorder=4)
            ox, oy, oyaw = rs.interpolate(raw_len, ctype, self.engine.max_curvature, ox, oy, oyaw)

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive RRT-Connect + Reeds-Shepp steering visualizer")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility (default: truly random each run)")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help=f"Turning radius (default {DEFAULT_RHO})")
    parser.add_argument("--range", dest="range_limit", type=float, default=DEFAULT_RANGE,
                         help=f"Max arc-length advanced per EXTEND (default {DEFAULT_RANGE})")
    args = parser.parse_args()
    App(seed=args.seed, rho=args.rho, range_limit=args.range_limit).show()


if __name__ == "__main__":
    main()
