# CS702 Assignment 2

## Prerequisites

This project uses [pixi](https://prefix.dev/) to manage the environment.
If you don't have pixi installed:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Install all dependencies (declared in `pixi.toml`):

```bash
pixi install
```

Dependencies include pygame and stljax for Problem 1 & 2, and mediapipe and opencv-python for Problem 3.

---

## Problem 1 — Flappy Bird Controller

### Run the game

```bash
pixi run python problem1/game.py
```

The game opens a pygame window. Use the keyboard to interact:

| Key   | Action |
|-------|--------|
| `M`   | Cycle mode: `manual` → `pid` → `mpc` → `human_in_loop` |
| `R`   | Reset the game |
| `SPACE` | Flap (manual mode) / human flap (human-in-loop mode) |
| `ESC` | Quit |

### Default mode

The game starts in **PID mode**. Because `PIDController.calc_input` is a stub
returning `0`, the bird will fall immediately. That is expected — implement
the controller to make it fly.

### What to implement

| Part | Location | What to do |
|------|----------|------------|
| 1.1 PID | [problem1/game.py:130](problem1/game.py#L130) | Fill in `PIDController.calc_input` |
| 1.2 MPC | [problem1/game.py:196](problem1/game.py#L196) | Fill in `MPCController._optimize` and `_cost` |
| 1.3 Human-in-loop | [problem1/game.py:265](problem1/game.py#L265) | Fill in `calculate_control_signal_human` |

Search for `# TODO` in `game.py` to find every stub.

### Quick smoke test (no display)

To verify the physics and controller skeleton run without errors in a headless
environment (e.g. CI, SSH):

```bash
pixi run python - <<'EOF'
from problem1.game import Bird, Pipe, PIDController, bird_motion, pipe_motion, calculate_control_signal
b, p, pid = Bird(), Pipe(), PIDController()
for _ in range(60):
    ctrl = calculate_control_signal(b, p, pid)
    bird_motion(b, ctrl, 1/60)
    pipe_motion(p, b, 1/60)
print("OK — bird y:", round(b.y, 1), " score-pipe x:", round(p.x, 1))
EOF
```

---

## Problem 2 — Trajectory Animation

All commands below should be run from the **`problem2/`** directory.

```bash
cd problem2
```

### 1. Preview the synthetic dataset

```bash
pixi run python -c "
from data import generate_dataset
trajs, hotspots = generate_dataset()
print('trajectories:', trajs.shape)
for h in hotspots:
    print(f'  hotspot {h.kind} at ({h.x:.0f}, {h.y:.0f}) t={h.time_step}')
"
```

### 2. Run the Part 1 pipeline

```bash
pixi run python part1.py
```

This generates synthetic data, runs bundling + layout (stubs initially, so
output = input), prints metrics, and opens the pygame viewer.

Options:

```bash
pixi run python part1.py --data path/to/trajs.npy   # custom data
pixi run python part1.py --clusters 4 --strength 0.5
pixi run python part1.py --record frames/part1/      # export PNGs
```

### 3. Run the Part 2 optimisation pipeline

```bash
pixi run python part2.py
```

This warm-starts from the Part 1 output, runs the JAX/Optax optimisation loop,
prints a comparison table (Baseline / Part 1 / Part 2), saves a convergence
plot to `reports/figures/convergence.png`, and opens the viewer.

Options:

```bash
pixi run python part2.py --steps 1000 --lr 0.005
pixi run python part2.py --record frames/part2/
pixi run python part2.py --save results/pos_opt.npy

# Ablation study — disable one loss term at a time:
pixi run python part2.py --ablate no_stl
pixi run python part2.py --ablate no_smooth
pixi run python part2.py --ablate no_deviation
pixi run python part2.py --ablate no_separation
```

### 4. One-shot demo

```bash
pixi run python run_demo.py
```

Generates data, runs Part 1 + Part 2 (200 steps), and plays back the result.

### 5. Export frames to video

```bash
pixi run python part2.py --record frames/part2/
ffmpeg -r 30 -i frames/part2/frame_%04d.png -pix_fmt yuv420p output.mp4
```

### What to implement

| Part | File | What to do |
|------|------|------------|
| 1 — Bundling & Layout | [part1.py](problem2/part1.py) | `pairwise_similarity`, `cluster_trajectories`, `bundle_trajectories`, `assign_offsets` |
| 2 — STL specs & Losses | [part2.py](problem2/part2.py) | `robustness_start_end`, `robustness_convergence`, `robustness_divergence`, `separation_loss` |

Search for `# TODO` to find every stub:

```bash
grep -n "# TODO" problem2/part1.py problem2/part2.py
```

---

## Problem 3 — Kinematics-based Hand Tracking

All commands below should be run from the **`problem3/`** directory.

```bash
cd problem3
```

### 1. Live capture + naïve kinematic prediction

```bash
pixi run python part1.py
```

Opens a webcam window showing the MediaPipe hand skeleton and prediction
circles at 100 ms, 200 ms, and 300 ms ahead of the index fingertip.
Press `q` or `Esc` to quit.

To record a session to file:

```bash
pixi run python part1.py --save data.npz
```

To replay saved data and evaluate RMSE:

```bash
pixi run python part1.py --load data.npz
```

### 2. Bayesian filtering

```bash
pixi run python part2.py --load data.npz   # offline RMSE comparison table
pixi run python part2.py --live --filter kf    # live KF overlay
pixi run python part2.py --live --filter ekf   # live EKF overlay
pixi run python part2.py --live --filter pf    # live particle filter overlay
pixi run python part2.py --live --filter mhe   # live MHE overlay
```

### What to implement

| Part | File | What to do |
|------|------|------------|
| 3.1 — Naïve prediction | [part1.py](problem3/part1.py) | `NaivePredictor.update`, `NaivePredictor.predict_ahead` |
| 3.2 — Kalman Filter | [filters.py](problem3/filters.py) | `KalmanFilter.predict`, `KalmanFilter.update` |
| 3.2 — EKF | [filters.py](problem3/filters.py) | `ExtendedKalmanFilter.predict`, `.update`, `.predict_ahead` |
| 3.2 — Particle Filter | [filters.py](problem3/filters.py) | `ParticleFilter.predict`, `.update`, `.predict_ahead` |
| 3.2 — MHE | [filters.py](problem3/filters.py) | `MovingHorizonEstimator.update` |

Search for `# TODO` to find every stub:

```bash
grep -n "# TODO\|NotImplementedError" problem3/part1.py problem3/filters.py
```
