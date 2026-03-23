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

The boilerplate for Problem 2 is in `\problem2`.

```bash
cd problem2
```

All the code that you have to implement are placed inside the `problem2.ipynb` file. Fill out the parts marked as `TODO`.


---

## Problem 3 

The boilerplate for Problem 3 is in `\problem3`.

```bash
cd problem3
```

Find the `problem3.ipynb` file. You should be able to answer all the questions (except for the open-ended challenge) by filling out the parts marked as `TODO`.