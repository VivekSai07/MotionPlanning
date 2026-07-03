"""
Interactive Reeds-Shepp steering visualizer.

This is the piece that answers "what does EXTEND() actually do for a
Reeds-Shepp car?" -- it is NOT straight-line interpolation. Given two SE(2)
poses q0=(x,y,theta) and q1=(x,y,theta) and a minimum turning radius rho,
the steering function:

  1. Enumerates all 12 Reeds-Shepp "word" formulas (LSL, LSR, LRL, LRLR, ...)
     under all 4 symmetry transforms (identity / timeflip / reflect / both)
     -> up to 48 candidate curves, each a short sequence of circular arcs
     (Left/Right turn at radius rho) and straight (Straight) segments,
     each with a signed length (negative = drive that segment backward).
  2. Keeps only the candidates that are geometrically valid for this
     (q0, q1, rho).
  3. Picks the candidate with the minimum total length -- this is exactly
     what OMPL's ReedsSheppStateSpace::getPath() does.
  4. Discretizes the winning path segment by segment and collision-checks
     every sampled point, exactly mirroring how CollisionFree() works in
     EXTEND(), just walking a curve instead of a line.

The core path-synthesis math (the 12 word functions, the symmetry
transforms, and the interpolation formulas) is ported from the
well-tested, MIT-licensed reference implementation in AtsushiSakai's
PythonRobotics (PathPlanning/ReedsSheppPath/reeds_shepp_path_planning.py),
itself implementing the formulas from:
    J.A. Reeds and L.A. Shepp, "Optimal paths for a car that goes both
    forwards and backwards," Pacific Journal of Mathematics, 145(2), 1990.
Ported (not merely wrapped) so this script has no external dependency
beyond matplotlib/numpy, and restructured to yield one micro-step at a
time for the interactive UI.

Pops up a matplotlib window with:
  - Left panel  : the world -- start/goal poses (with an illustrative
                  car-footprint rectangle), the turning circle for the
                  segment currently being traced, and collision-check
                  sample points.
  - Upper-right : the steering procedure's conceptual steps, with the
                  current one highlighted (enumerate -> pick shortest ->
                  discretize -> collision-check -> return).
  - Lower-right : "Function Detail" -- the full candidate table (all
                  valid RS words and their lengths, shortest highlighted),
                  then the turning-circle / arc math for whichever segment
                  is currently being traced.
  - Buttons     : "<< Back" / "Next >>" / "Reset (new random poses)".
                  (Left/Right arrow keys and 'r' also work.)

Run:
    python reeds_shepp_interactive.py
    python reeds_shepp_interactive.py --seed 3 --rho 2.0
    python reeds_shepp_interactive.py --start 1 1 0 --goal 9 9 90
"""

import argparse
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

# ----------------------------------------------------------------------
# World setup (same map used throughout this series of examples)
# ----------------------------------------------------------------------
BOUNDS = (0.0, 10.0)
OBSTACLE = dict(x=5.0, y=4.0, w=2.0, h=2.0)
DEFAULT_START = (1.0, 1.0, 0.0)             # facing +x
DEFAULT_GOAL = (9.0, 9.0, math.pi / 2)      # facing +y
DEFAULT_RHO = 2.0

# Illustrative Opel Corsa F footprint (approximate public dimensions,
# for visual flavor only -- not the deliverable's collision model, which
# here simplifies the car to its reference point).
CAR_LENGTH = 4.06
CAR_WIDTH = 1.77

SAMPLES_PER_SEGMENT = 40
CANDIDATES_SHOWN = 10


# ----------------------------------------------------------------------
# Reeds-Shepp core math (ported from PythonRobotics, see module docstring)
# ----------------------------------------------------------------------
def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def mod2pi(x):
    v = math.fmod(x, 2.0 * math.pi if x >= 0 else -2.0 * math.pi)
    if v < -math.pi:
        v += 2.0 * math.pi
    elif v > math.pi:
        v -= 2.0 * math.pi
    return v


def polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


def left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ['L', 'S', 'L']
    return False, [], []


def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, v], ['L', 'S', 'R']
    return False, [], []


def left_x_right_x_left(x, y, phi):
    zeta, eeta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = mod2pi(a + theta + math.pi / 2)
        u = mod2pi(math.pi - 2 * a)
        v = mod2pi(phi - t - u)
        return True, [t, -u, v], ['L', 'R', 'L']
    return False, [], []


def left_x_right_left(x, y, phi):
    zeta, eeta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = mod2pi(a + theta + math.pi / 2)
        u = mod2pi(math.pi - 2 * a)
        v = mod2pi(-phi + t + u)
        return True, [t, -u, -v], ['L', 'R', 'L']
    return False, [], []


def left_right_x_left(x, y, phi):
    zeta, eeta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 <= 4.0:
        u = math.acos(1 - u1 ** 2 * 0.125)
        a = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(-a + theta + math.pi / 2)
        v = mod2pi(t - u - phi)
        return True, [t, u, -v], ['L', 'R', 'L']
    return False, [], []


def left_right_x_left_right(x, y, phi):
    zeta, eeta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 <= 2:
        a = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + a + math.pi / 2)
        u = mod2pi(a)
        v = mod2pi(phi - t + 2 * u)
        if t >= 0 and u >= 0 and v >= 0:
            return True, [t, u, -u, -v], ['L', 'R', 'L', 'R']
    return False, [], []


def left_x_right_left_x_right(x, y, phi):
    zeta, eeta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1 ** 2) / 16
    if 0 <= u2 <= 1:
        u = math.acos(u2)
        a = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(theta + a + math.pi / 2)
        v = mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return True, [t, -u, -u, v], ['L', 'R', 'L', 'R']
    return False, [], []


def left_x_right90_straight_left(x, y, phi):
    zeta, eeta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 >= 2.0:
        u = math.sqrt(u1 ** 2 - 4) - 2
        a = math.atan2(2, math.sqrt(u1 ** 2 - 4))
        t = mod2pi(theta + a + math.pi / 2)
        v = mod2pi(t - phi + math.pi / 2)
        if t >= 0 and v >= 0:
            return True, [t, -math.pi / 2, -u, -v], ['L', 'R', 'S', 'L']
    return False, [], []


def left_straight_right90_x_left(x, y, phi):
    zeta, eeta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 >= 2.0:
        u = math.sqrt(u1 ** 2 - 4) - 2
        a = math.atan2(math.sqrt(u1 ** 2 - 4), 2)
        t = mod2pi(theta - a + math.pi / 2)
        v = mod2pi(t - phi - math.pi / 2)
        if t >= 0 and v >= 0:
            return True, [t, u, math.pi / 2, -v], ['L', 'S', 'R', 'L']
    return False, [], []


def left_x_right90_straight_right(x, y, phi):
    zeta, eeta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 >= 2.0:
        t = mod2pi(theta + math.pi / 2)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi / 2)
        if t >= 0 and v >= 0:
            return True, [t, -math.pi / 2, -u, -v], ['L', 'R', 'S', 'R']
    return False, [], []


def left_straight_left90_x_right(x, y, phi):
    zeta, eeta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi / 2)
        if t >= 0 and v >= 0:
            return True, [t, u, math.pi / 2, -v], ['L', 'S', 'L', 'R']
    return False, [], []


def left_x_right90_straight_left90_x_right(x, y, phi):
    zeta, eeta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    if u1 >= 4.0:
        u = math.sqrt(u1 ** 2 - 4) - 4
        a = math.atan2(2, math.sqrt(u1 ** 2 - 4))
        t = mod2pi(theta + a + math.pi / 2)
        v = mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return True, [t, -math.pi / 2, -u, -math.pi / 2, v], ['L', 'R', 'S', 'L', 'R']
    return False, [], []


PATH_FUNCTIONS = [
    left_straight_left, left_straight_right,                                   # CSC
    left_x_right_x_left, left_x_right_left, left_right_x_left,                 # CCC
    left_right_x_left_right, left_x_right_left_x_right,                        # CCCC
    left_x_right90_straight_left, left_x_right90_straight_right,               # CCSC
    left_straight_right90_x_left, left_straight_left90_x_right,                # CSCC
    left_x_right90_straight_left90_x_right,                                    # CCSCC
]


def timeflip(lengths):
    return [-x for x in lengths]


def reflect(ctypes):
    swap = {'L': 'R', 'R': 'L', 'S': 'S'}
    return [swap[c] for c in ctypes]


def generate_candidates(q0, q1, max_curvature):
    """Enumerate every valid Reeds-Shepp candidate for q0->q1. Returns a
    list of dicts: {lengths (curvature-normalized, signed), ctypes,
    total_len (real units)}."""
    dx, dy, dth = q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]
    c, s = math.cos(q0[2]), math.sin(q0[2])
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature

    candidates = []
    seen = []

    def try_add(lengths, ctypes):
        total = sum(abs(v) for v in lengths)
        if total <= 1e-6:
            return
        key = (tuple(ctypes), round(total, 3))
        for k_ct, k_tot in seen:
            if k_ct == key[0] and abs(k_tot - key[1]) < 0.05:
                return
        seen.append(key)
        candidates.append(dict(lengths=lengths, ctypes=ctypes,
                                total_len=total / max_curvature))

    for fn in PATH_FUNCTIONS:
        ok, lengths, ctypes = fn(x, y, dth)
        if ok:
            try_add(lengths, ctypes)

        ok, lengths, ctypes = fn(-x, y, -dth)
        if ok:
            try_add(timeflip(lengths), ctypes)

        ok, lengths, ctypes = fn(x, -y, -dth)
        if ok:
            try_add(lengths, reflect(ctypes))

        ok, lengths, ctypes = fn(-x, -y, dth)
        if ok:
            try_add(timeflip(lengths), reflect(ctypes))

    candidates.sort(key=lambda c: c["total_len"])
    return candidates


def interpolate(dist, mode, max_curvature, ox, oy, oyaw):
    """Pose reached after driving `dist` (signed, curvature-normalized)
    along a segment of type `mode` starting at (ox,oy,oyaw)."""
    if mode == "S":
        x = ox + dist / max_curvature * math.cos(oyaw)
        y = oy + dist / max_curvature * math.sin(oyaw)
        yaw = oyaw
    else:
        ldx = math.sin(dist) / max_curvature
        if mode == "L":
            ldy = (1.0 - math.cos(dist)) / max_curvature
            yaw = oyaw + dist
        else:  # "R"
            ldy = (1.0 - math.cos(dist)) / -max_curvature
            yaw = oyaw - dist
        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy
        x, y = ox + gdx, oy + gdy
    return x, y, yaw


def turning_center(x, y, yaw, rho, mode):
    """Center of the circle of radius rho that segment `mode` ('L'/'R')
    is driving along, starting from pose (x,y,yaw)."""
    if mode == "L":
        return (x - rho * math.sin(yaw), y + rho * math.cos(yaw))
    else:
        return (x + rho * math.sin(yaw), y - rho * math.cos(yaw))


# ----------------------------------------------------------------------
# World / collision helpers
# ----------------------------------------------------------------------
def obstacle_contains(p):
    x, y = p
    return (OBSTACLE["x"] <= x <= OBSTACLE["x"] + OBSTACLE["w"] and
            OBSTACLE["y"] <= y <= OBSTACLE["y"] + OBSTACLE["h"])


def car_corners(x, y, yaw, length=CAR_LENGTH, width=CAR_WIDTH):
    hl, hw = length / 2, width / 2
    local = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * lx - s * ly, y + s * lx + c * ly) for (lx, ly) in local]


CONCEPT_LINES = [
    "Transform (q0, q1) into q0's local frame (origin, heading = 0)",
    "Try 12 RS word types x 4 symmetry variants -> up to 48 candidates",
    "Keep only the geometrically valid candidates",
    "Pick the candidate with minimum total length  (OMPL: getPath())",
    "Discretize the winning path, segment by segment  (interpolate())",
    "Collision-check every sampled (x,y) along the path",
    "Return path  (Trapped if any sample collides)",
]


# ----------------------------------------------------------------------
# Engine: generator that yields UI snapshots one micro-step at a time.
# ----------------------------------------------------------------------
class Engine:
    def __init__(self, q0, q1, rho):
        self.q0 = q0
        self.q1 = q1
        self.rho = rho
        self.max_curvature = 1.0 / rho
        self.winner = None
        self.trapped_at = None
        self.full_samples = []  # list of (x,y,yaw,direction,seg_index)

    def _base(self, **kw):
        snap = dict(q0=self.q0, q1=self.q1, rho=self.rho,
                    title="", info="", info_color="black", detail="",
                    full_samples=list(self.full_samples))
        snap.update(kw)
        return snap

    def run(self):
        yield self._base(
            type="init", highlight=[1],
            title="Setup",
            info=f"q0=({self.q0[0]:.1f},{self.q0[1]:.1f},{math.degrees(self.q0[2]):.0f}°)   "
                 f"q1=({self.q1[0]:.1f},{self.q1[1]:.1f},{math.degrees(self.q1[2]):.0f}°)   rho={self.rho}",
            detail=(f"q0 = {tuple(round(v, 2) for v in self.q0)}\n"
                    f"q1 = {tuple(round(v, 2) for v in self.q1)}\n"
                    f"rho (turning radius) = {self.rho}\n"
                    f"max_curvature = 1/rho = {self.max_curvature:.3f}\n\n"
                    "Local frame: translate so q0 -> origin, rotate so\n"
                    "q0's heading -> 0. All 12 word formulas are solved\n"
                    "in this normalized frame (same trick as the paper)."),
        )

        candidates = generate_candidates(self.q0, self.q1, self.max_curvature)
        if not candidates:
            yield self._base(type="no_candidates", highlight=[2, 3],
                              title="No candidate found",
                              info="No valid Reeds-Shepp path for this configuration.",
                              info_color="red",
                              detail="generate_candidates() returned nothing -- try Reset.")
            return None

        lines = [f"Candidates found: {len(candidates)}  (showing shortest {min(CANDIDATES_SHOWN, len(candidates))})"]
        for i, c in enumerate(candidates[:CANDIDATES_SHOWN]):
            label = " ".join(f"{t}{'+' if l >= 0 else '-'}" for t, l in zip(c["ctypes"], c["lengths"]))
            marker = "  <- SHORTEST" if i == 0 else ""
            lines.append(f"  [{label:<18s}] length={c['total_len']:.2f}{marker}")
        if len(candidates) > CANDIDATES_SHOWN:
            lines.append(f"  ... and {len(candidates) - CANDIDATES_SHOWN} more candidate(s)")
        yield self._base(
            type="candidates", highlight=[2, 3], candidates=candidates,
            title=f"Enumerate candidates ({len(candidates)} valid)",
            info=f"{len(candidates)} valid Reeds-Shepp candidates found",
            detail="\n".join(lines),
        )

        winner = candidates[0]
        self.winner = winner
        seg_desc = " ".join(f"{t}{'+' if l >= 0 else '-'}({abs(l) * self.rho:.2f}m)"
                             for t, l in zip(winner["ctypes"], winner["lengths"]))
        yield self._base(
            type="winner", highlight=[4], candidates=candidates,
            title="Pick shortest candidate",
            info=f"Winner: {seg_desc}   total={winner['total_len']:.2f}",
            info_color="darkgreen",
            detail=(f"Shortest candidate:\n"
                    f"  ctypes  = {winner['ctypes']}\n"
                    f"  signs   = {['forward' if l >= 0 else 'backward' for l in winner['lengths']]}\n"
                    f"  total length = {winner['total_len']:.2f} (real units, rho={self.rho})\n\n"
                    "'+' = forward, '-' = backward, per Reeds-Shepp convention\n"
                    "(negative signed length = drive that arc/line in reverse)."),
        )

        ox, oy, oyaw = 0.0, 0.0, 0.0  # local frame, matches generate_candidates' frame
        trapped = False
        for seg_i, (ctype, raw_len) in enumerate(zip(winner["ctypes"], winner["lengths"])):
            real_len = abs(raw_len) * self.rho
            direction = "forward" if raw_len >= 0 else "backward"

            start_global = self._to_global(ox, oy, oyaw)
            if ctype in ("L", "R"):
                center_local = turning_center(ox, oy, oyaw, self.rho, ctype)
                center_global = self._to_global(*center_local, 0.0)[:2]
                detail = (f"Segment {seg_i + 1}/{len(winner['ctypes'])}: {ctype}-turn, {direction}\n"
                          f"  arc angle = {abs(raw_len):.3f} rad ({math.degrees(abs(raw_len)):.1f}°)\n"
                          f"  arc length = angle * rho = {real_len:.2f} m\n"
                          f"  turning center = pos + rho*(perp. to heading)\n"
                          f"                 = ({center_global[0]:.2f}, {center_global[1]:.2f})")
            else:
                center_global = None
                detail = (f"Segment {seg_i + 1}/{len(winner['ctypes'])}: Straight, {direction}\n"
                          f"  length = {real_len:.2f} m")

            yield self._base(
                type="segment_setup", highlight=[5], seg_index=seg_i,
                seg_ctype=ctype, seg_direction=direction, center=center_global,
                pose_start=start_global,
                title=f"Segment {seg_i + 1}: {ctype} ({direction})",
                info=f"{ctype}-segment, {direction}, length={real_len:.2f}m",
                detail=detail,
            )

            # discretize + collision-check this segment (in local frame,
            # transform each sample to global as we go)
            seg_samples = []
            collided_at = None
            n = SAMPLES_PER_SEGMENT
            for i in range(n + 1):
                dist = raw_len * i / n
                lx, ly, lyaw = interpolate(dist, ctype, self.max_curvature, ox, oy, oyaw)
                gx, gy, gyaw = self._to_global(lx, ly, lyaw)
                inside = obstacle_contains((gx, gy))
                seg_samples.append((gx, gy, gyaw, inside))
                if inside and collided_at is None:
                    collided_at = (gx, gy)
                    break

            ok = collided_at is None
            ox0, oy0 = OBSTACLE["x"], OBSTACLE["y"]
            ox1, oy1 = ox0 + OBSTACLE["w"], oy0 + OBSTACLE["h"]
            if ok:
                coll_detail = (f"Collision check ({n + 1} samples) along this segment:\n"
                                f"  obstacle = x:[{ox0:.0f},{ox1:.0f}], y:[{oy0:.0f},{oy1:.0f}]\n"
                                f"  every sampled (x,y) is outside the obstacle\n"
                                f"  => segment collision-free")
            else:
                coll_detail = (f"Collision check along this segment:\n"
                                f"  obstacle = x:[{ox0:.0f},{ox1:.0f}], y:[{oy0:.0f},{oy1:.0f}]\n"
                                f"  hit at ({collided_at[0]:.2f}, {collided_at[1]:.2f})\n"
                                f"  => COLLISION: path is Trapped here")
            yield self._base(
                type="segment_collision", highlight=[6], seg_index=seg_i,
                seg_ctype=ctype, center=center_global, seg_samples=seg_samples, ok=ok,
                title=f"Segment {seg_i + 1}: Collision Check",
                info=("collision-free" if ok else "COLLISION -> Trapped"),
                info_color=("green" if ok else "red"),
                detail=coll_detail,
            )

            if not ok:
                trapped = True
                self.trapped_at = collided_at
                break

            # commit this segment's samples (without the inside flag noise)
            for (gx, gy, gyaw, _inside) in seg_samples:
                self.full_samples.append((gx, gy, gyaw, raw_len >= 0, seg_i))
            ox, oy, oyaw = interpolate(raw_len, ctype, self.max_curvature, ox, oy, oyaw)

        if trapped:
            yield self._base(
                type="trapped", highlight=[7],
                title="Return: Trapped",
                info="Collision along the Reeds-Shepp curve -> EXTEND would report Trapped.",
                info_color="red",
                detail=("Return Trapped;\n"
                        "  In RRT-Connect's EXTEND(), this candidate steer\n"
                        "  result is discarded -- no vertex is added, and\n"
                        "  a new q_rand / nearest-neighbor pair is tried next."),
            )
            return None

        yield self._base(
            type="complete", highlight=[7],
            title="Return path -- Reeds-Shepp steering complete",
            info=f"Full path collision-free. Total length = {winner['total_len']:.2f}",
            info_color="darkgreen",
            detail=("Return path;\n"
                    f"  {len(winner['ctypes'])} segment(s), total length = {winner['total_len']:.2f}\n\n"
                    "This whole curve is what EXTEND() would use in place\n"
                    "of a straight q_near -> q_new edge for a Reeds-Shepp\n"
                    "vehicle -- collision-checked exactly like before, just\n"
                    "walking an arc/line sequence instead of one line."),
        )
        return self.q1

    def _to_global(self, lx, ly, lyaw):
        c, s = math.cos(self.q0[2]), math.sin(self.q0[2])
        gx = c * lx - s * ly + self.q0[0]
        gy = s * lx + c * ly + self.q0[1]
        gyaw = pi_2_pi(lyaw + self.q0[2])
        return gx, gy, gyaw


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
class App:
    def __init__(self, seed, rho, start, goal):
        self.seed = seed
        self.rho = rho
        self.fixed_start = start
        self.fixed_goal = goal
        self.history = []
        self.cursor = -1
        self._new_run()

        self.fig = plt.figure(figsize=(15, 8.5))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.9, 1.2], height_ratios=[0.55, 1.45],
                                    left=0.045, right=0.98, top=0.94, bottom=0.15,
                                    wspace=0.14, hspace=0.2)
        self.ax_world = self.fig.add_subplot(gs[:, 0])
        self.ax_code = self.fig.add_subplot(gs[0, 1])
        self.ax_detail = self.fig.add_subplot(gs[1, 1])

        ax_back = self.fig.add_axes([0.20, 0.03, 0.15, 0.055])
        ax_next = self.fig.add_axes([0.37, 0.03, 0.15, 0.055])
        ax_reset = self.fig.add_axes([0.54, 0.03, 0.24, 0.055])
        self.btn_back = Button(ax_back, "<< Back")
        self.btn_next = Button(ax_next, "Next >>")
        self.btn_reset = Button(ax_reset, "Reset (new random poses)")
        self.btn_back.on_clicked(self.on_back)
        self.btn_next.on_clicked(self.on_next)
        self.btn_reset.on_clicked(self.on_reset)

        self.status_text = self.fig.text(0.045, 0.10, "", fontsize=9, family="monospace")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.on_next(None)

    def _new_run(self):
        if self.fixed_start is not None and self.fixed_goal is not None:
            q0, q1 = self.fixed_start, self.fixed_goal
        else:
            actual_seed = self.seed if self.seed is not None else random.SystemRandom().randrange(2**31)
            print(f"Starting new Reeds-Shepp run with seed={actual_seed}")
            rng = random.Random(actual_seed)
            q0 = (rng.uniform(*BOUNDS), rng.uniform(*BOUNDS), rng.uniform(-math.pi, math.pi))
            q1 = (rng.uniform(*BOUNDS), rng.uniform(*BOUNDS), rng.uniform(-math.pi, math.pi))
        self.engine = Engine(q0, q1, self.rho)
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
        self.fixed_start = None
        self.fixed_goal = None
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
        pad = 3.0
        ax.set_xlim(BOUNDS[0] - pad, BOUNDS[1] + pad)
        ax.set_ylim(BOUNDS[0] - pad, BOUNDS[1] + pad)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)
        rect = patches.Rectangle((OBSTACLE["x"], OBSTACLE["y"]), OBSTACLE["w"], OBSTACLE["h"],
                                  facecolor="0.25", edgecolor="black", zorder=2)
        ax.add_patch(rect)

    def _draw_pose(self, ax, pose, color, label, footprint=True):
        x, y, yaw = pose
        if footprint:
            poly = patches.Polygon(car_corners(x, y, yaw), closed=True,
                                    fill=False, edgecolor=color, linewidth=1.3,
                                    linestyle=":", zorder=4)
            ax.add_patch(poly)
        ax.annotate("", xy=(x + 1.4 * math.cos(yaw), y + 1.4 * math.sin(yaw)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5), zorder=6)
        ax.plot(x, y, "o", color=color, markersize=6, zorder=6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 10), color=color, fontweight="bold")

    def render(self):
        snap = self.history[self.cursor]
        ax = self.ax_world
        ax.clear()
        self._draw_static_world()

        # committed path so far, colored by direction
        fs = snap["full_samples"]
        if fs:
            seg_ids = sorted(set(p[4] for p in fs))
            for sid in seg_ids:
                pts = [p for p in fs if p[4] == sid]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                fwd = pts[0][3]
                ax.plot(xs, ys, "-" if fwd else "--", color="steelblue" if fwd else "crimson",
                        linewidth=2.5, zorder=5)

        t = snap["type"]
        if t == "segment_setup":
            ax.plot(*snap["pose_start"][:2], "o", color="darkviolet", markersize=8, zorder=6)
            if snap.get("center") is not None:
                cx, cy = snap["center"]
                circ = patches.Circle((cx, cy), snap["rho"], fill=False, linestyle="--",
                                       color="darkviolet", linewidth=1.3, zorder=3)
                ax.add_patch(circ)
                ax.plot(cx, cy, "+", color="darkviolet", markersize=10, mew=2, zorder=6)
        elif t == "segment_collision":
            if snap.get("center") is not None:
                cx, cy = snap["center"]
                circ = patches.Circle((cx, cy), self.engine.rho, fill=False, linestyle="--",
                                       color="darkviolet", alpha=0.5, linewidth=1.3, zorder=3)
                ax.add_patch(circ)
            for (gx, gy, _gyaw, inside) in snap.get("seg_samples", []):
                ax.plot(gx, gy, ".", color=("red" if inside else "gray"),
                        markersize=7 if inside else 4, alpha=0.9 if inside else 0.6, zorder=6)
            color = "green" if snap["ok"] else "red"
            xs = [p[0] for p in snap.get("seg_samples", [])]
            ys = [p[1] for p in snap.get("seg_samples", [])]
            ax.plot(xs, ys, "-", color=color, alpha=0.6, linewidth=2, zorder=5)

        self._draw_pose(ax, snap["q0"], "seagreen", "Start (q0)")
        self._draw_pose(ax, snap["q1"], "firebrick", "Goal (q1)")

        if self.engine.trapped_at and t in ("trapped",):
            ax.plot(*self.engine.trapped_at, "x", color="darkred", markersize=18, mew=4, zorder=7)

        ax.set_title(snap["title"])

        # concept-steps panel
        axc = self.ax_code
        axc.clear()
        axc.axis("off")
        axc.set_xlim(0, 1)
        axc.set_ylim(0, len(CONCEPT_LINES) + 1)
        axc.set_title("Reeds-Shepp Steering -- Conceptual Steps")
        hl = snap.get("highlight", [])
        for i, line in enumerate(CONCEPT_LINES, start=1):
            y = len(CONCEPT_LINES) - i + 1
            if i in hl:
                axc.add_patch(patches.Rectangle((0, y - 0.42), 1, 0.84,
                                                  color="yellow", alpha=0.5, zorder=0))
            axc.text(0.02, y, f"{i}. {line}", fontsize=9, family="monospace", va="center", zorder=1)

        # function detail panel
        axd = self.ax_detail
        axd.clear()
        axd.axis("off")
        axd.set_xlim(0, 1)
        axd.set_ylim(0, 1)
        axd.set_title("Function Detail")
        axd.text(0.02, 0.97, snap.get("detail", ""), fontsize=8.6, family="monospace",
                  va="top", ha="left", transform=axd.transAxes)

        self.status_text.set_text(
            f"Step {self.cursor + 1}/{len(self.history)}   |   {snap['info']}"
        )
        self.status_text.set_color(snap.get("info_color", "black"))

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def parse_pose(vals):
    x, y, theta_deg = vals
    return (float(x), float(y), math.radians(float(theta_deg)))


def main():
    parser = argparse.ArgumentParser(description="Interactive Reeds-Shepp steering visualizer")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducible random start/goal poses")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help=f"Turning radius (default {DEFAULT_RHO})")
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "THETA_DEG"), default=None,
                         help="Fixed start pose (default: (1,1,0))")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "THETA_DEG"), default=None,
                         help="Fixed goal pose (default: (9,9,90))")
    args = parser.parse_args()

    if args.start is None and args.goal is None and args.seed is None:
        start, goal = DEFAULT_START, DEFAULT_GOAL
    else:
        start = parse_pose(args.start) if args.start else DEFAULT_START
        goal = parse_pose(args.goal) if args.goal else DEFAULT_GOAL
        if args.seed is not None and args.start is None and args.goal is None:
            start, goal = None, None  # let seed drive random poses

    App(seed=args.seed, rho=args.rho, start=start, goal=goal).show()


if __name__ == "__main__":
    main()
