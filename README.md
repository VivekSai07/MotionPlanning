# Motion Planning Visualizers

Step-by-step, interactive visualizations built while working through the
sampling-based motion planning sequence for the *Advanced Mobile Robotics*
course (University of Stuttgart, SS26): **RRT → RRT-Connect → Reeds-Shepp
steering** — the last of which is the local planner an SE(2) `EXTEND()`
needs before this can be wired into an actual OMPL `RRTConnect` planner.

Every script uses real randomness / real geometry (no scripted or
pre-baked outputs) and lets you click through the algorithm one micro-step
at a time, seeing exactly what each internal function (`Nearest()`,
`Steer()`/`NEW_CONFIG()`, `CollisionFree()`, ...) computes along the way.

![RRT tree growth animation](media/rrt_animation.gif)

## Contents

| Script | What it shows |
|---|---|
| `rrt_visualization.py` | A fixed, hand-worked-through RRT example (matplotlib animation + static summary PNG) — reproduces a specific walkthrough exactly, node for node, to sanity-check the algorithm by hand first. |
| `rrt_interactive.py` | Interactive RRT: a real-time, click-through (`Next`/`Back`) run with genuine `random.uniform()` sampling, a live pseudocode highlight, and a "Function Detail" panel breaking down `Nearest()`, `Steer()`, and `CollisionFree()` down to the arithmetic. |
| `rrt_connect_interactive.py` | Interactive **RRT-Connect** (Kuffner & LaValle, 2000): two trees, `EXTEND()` / `CONNECT()` / `SWAP()`, with the pseudocode call-stack highlighted across all three published pseudocode blocks as execution moves between them. |
| `reeds_shepp_interactive.py` | Interactive **Reeds-Shepp steering**: enumerates all 12 RS word types × 4 symmetry variants (≈48 candidate curves), picks the shortest (mirroring `OMPL::ReedsSheppStateSpace::getPath()`), then discretizes and collision-checks it segment by segment — this is what a real `EXTEND()` uses in place of straight-line interpolation for a car-like (SE(2)) robot. |

## Demo

| RRT final tree + path | RRT-Connect / Reeds-Shepp |
|---|---|
| ![RRT final tree](media/rrt_final_tree.png) | run the interactive scripts below — they pop up a live matplotlib window, which doesn't screenshot well as a static image |

## Requirements

```
python >= 3.9
matplotlib
numpy
pillow      # only needed for GIF export in rrt_visualization.py
```

Install with:

```bash
pip install matplotlib numpy pillow
```

## Usage

Each interactive script pops up a matplotlib window with `<< Back` /
`Next >>` / `Reset` buttons (arrow keys and `r` also work).

```bash
# Fixed, worked-through example -> saves media/rrt_animation.gif + rrt_final_tree.png
python rrt_visualization.py

# Interactive RRT (pure-uniform sampling by default; toggle goal-bias in the UI)
python rrt_interactive.py
python rrt_interactive.py --seed 42

# Interactive RRT-Connect (two trees, real pseudocode call-stack highlighting)
python rrt_connect_interactive.py
python rrt_connect_interactive.py --seed 1

# Interactive Reeds-Shepp steering
python reeds_shepp_interactive.py
python reeds_shepp_interactive.py --seed 5 --rho 2.0
python reeds_shepp_interactive.py --start 1 1 0 --goal 9 2 0
```

## Notes on correctness

- All four scripts share the same 10×10 world, obstacle, start `(1,1)`,
  and goal `(9,9)` used throughout, for continuity across the series.
- The RRT / RRT-Connect collision checks, nearest-neighbor search, and
  steering are implemented directly (no external planning library).
- The Reeds-Shepp path-synthesis formulas (all 12 word types) are ported
  from the well-tested, MIT-licensed reference implementation in
  [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
  (`PathPlanning/ReedsSheppPath/reeds_shepp_path_planning.py`), itself
  implementing:
  > J.A. Reeds and L.A. Shepp, "Optimal paths for a car that goes both
  > forwards and backwards," *Pacific Journal of Mathematics*, 145(2),
  > 1990.

  The port was independently verified: driving the synthesized path from
  a start pose must land exactly on the goal pose. Across 200 random
  `(start, goal, turning-radius)` trials, max reconstruction error was
  `~4e-15` (floating-point noise).
- The Reeds-Shepp visualizer treats the vehicle as a point for collision
  checking (an illustrative Opel Corsa F footprint rectangle is drawn at
  the start/goal poses for scale only). Sweeping the full oriented
  rectangle along the curve is a further step, not yet implemented here.

## License

MIT — see [LICENSE](LICENSE).
