"""Configuration and shared types for Problem 2: Trajectory Animation."""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data dimensions
# ---------------------------------------------------------------------------
N: int = 20  # number of objects / trajectories
K: int = 60  # number of time steps (after resampling)

# ---------------------------------------------------------------------------
# Window / rendering
# ---------------------------------------------------------------------------
WINDOW_WIDTH: int = 800
WINDOW_HEIGHT: int = 600
FPS: int = 30
DOT_RADIUS: int = 4
BG_COLOR = (255, 255, 255)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Optimisation (Part 2)
# ---------------------------------------------------------------------------
N_STEPS: int = 500  # gradient descent iterations
LEARNING_RATE: float = 1e-2
LOG_EVERY: int = 50  # print/log every N steps

# ---------------------------------------------------------------------------
# Weights for the combined loss (Part 2)
# ---------------------------------------------------------------------------
W_STL: float = 1.0  # STL robustness (negated, to maximise)
W_SMOOTH: float = 0.1  # smoothness penalty
W_DEVIATION: float = 0.5  # deviation from input trajectories
W_SEPARATION: float = 0.2  # pairwise separation / anti-occlusion

# ---------------------------------------------------------------------------
# Hotspot tolerances
# ---------------------------------------------------------------------------
CONVERGE_RADIUS: float = 30.0  # pixels — objects must be within this of hotspot
DIVERGE_RADIUS: float = 30.0
HOTSPOT_TIME_WINDOW: tuple[int, int] = (K // 3, 2 * K // 3)  # frame range


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


@dataclass
class Hotspot:
    x: float
    y: float
    kind: str  # "converge" | "diverge"
    group: list[int]  # indices of objects belonging to this hotspot
    time_step: int  # frame at which the hotspot is active


@dataclass
class Hotspot3D:
    x: float
    y: float
    z: float
    kind: str  # "converge" | "diverge"
    group: list[int]  # indices of objects belonging to this hotspot
    time_step: int  # frame at which the hotspot is active
