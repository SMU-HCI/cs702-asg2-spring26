"""Synthetic dataset generation and trajectory loaders."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from config import N, K, WINDOW_WIDTH, WINDOW_HEIGHT, RANDOM_SEED, Hotspot
from utils import resample_all


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def generate_dataset(
    n: int = N,
    k: int = K,
    seed: int = RANDOM_SEED,
    width: int = WINDOW_WIDTH,
    height: int = WINDOW_HEIGHT,
) -> tuple[NDArray, list[Hotspot]]:
    """Generate N synthetic trajectories of length K with two hotspots.

    Scenario:
      - All objects start from random positions on the LEFT side.
      - They converge toward a central hotspot at ~K/3.
      - They then diverge toward random endpoints on the RIGHT side.

    Args:
        n:      Number of trajectories.
        k:      Number of time steps.
        seed:   Random seed for reproducibility.
        width:  Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        traj_in:  Shape (N, K, 2) — input trajectories.
        hotspots: List of Hotspot objects describing the dataset structure.
    """
    rng = np.random.default_rng(seed)

    # --- Define hotspot positions ---
    converge_pt = np.array([width * 0.4, height * 0.5], dtype=np.float32)
    diverge_pt  = np.array([width * 0.6, height * 0.5], dtype=np.float32)

    t_converge = k // 3      # frame at which objects arrive at converge hotspot
    t_diverge  = 2 * k // 3  # frame at which objects leave diverge hotspot

    # --- Generate start / end positions ---
    starts = rng.uniform([0, 0], [width * 0.15, height], size=(n, 2)).astype(np.float32)
    ends   = rng.uniform([width * 0.85, 0], [width, height], size=(n, 2)).astype(np.float32)

    # --- Build trajectories: linear segments with added noise ---
    trajs = np.zeros((n, k, 2), dtype=np.float32)

    # Add slight per-object offset around the hotspots to avoid perfect overlap.
    offset_c = rng.normal(0, 8, size=(n, 2)).astype(np.float32)
    offset_d = rng.normal(0, 8, size=(n, 2)).astype(np.float32)
    converge_pts = converge_pt + offset_c
    diverge_pts  = diverge_pt  + offset_d

    # Segment keyframes for each object: start -> converge -> diverge -> end
    waypoints_t = [0, t_converge, t_diverge, k - 1]

    for i in range(n):
        wps = np.stack([starts[i], converge_pts[i], diverge_pts[i], ends[i]])  # (4, 2)
        for seg in range(len(waypoints_t) - 1):
            t0, t1 = waypoints_t[seg], waypoints_t[seg + 1]
            steps = t1 - t0
            for d in range(2):
                trajs[i, t0:t1, d] = np.linspace(wps[seg, d], wps[seg + 1, d], steps)
        trajs[i, -1] = ends[i]

    # Add small smooth noise.
    from scipy.ndimage import gaussian_filter1d
    noise = rng.normal(0, 3, size=trajs.shape).astype(np.float32)
    noise = gaussian_filter1d(noise, sigma=3, axis=1)
    trajs += noise
    trajs = np.clip(trajs, 0, [[width, height]])

    # --- Define hotspots ---
    hotspots = [
        Hotspot(
            x=float(converge_pt[0]),
            y=float(converge_pt[1]),
            kind="converge",
            group=list(range(n)),
            time_step=t_converge,
        ),
        Hotspot(
            x=float(diverge_pt[0]),
            y=float(diverge_pt[1]),
            kind="diverge",
            group=list(range(n)),
            time_step=t_diverge,
        ),
    ]

    return trajs, hotspots


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_trajectories(path: str | Path, resample_to: int = K) -> NDArray:
    """Load trajectories from a .npy file and resample to ``resample_to`` steps.

    Expected on-disk format: shape (N, T, 2) as a NumPy float32 array.

    Args:
        path:        Path to the .npy file.
        resample_to: Number of time steps to resample to.

    Returns:
        Shape (N, resample_to, 2) float32 array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    trajs = np.load(path).astype(np.float32)

    if trajs.ndim != 3 or trajs.shape[2] != 2:
        raise ValueError(
            f"Expected shape (N, T, 2), got {trajs.shape}. "
            "Each trajectory must be a sequence of 2-D points."
        )

    if trajs.shape[1] != resample_to:
        trajs = resample_all(trajs, resample_to)

    return trajs


def save_positions(pos: NDArray, path: str | Path) -> None:
    """Save output positions to a .npy file.

    Args:
        pos:  Shape (K, N, 2).
        path: Destination path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pos.astype(np.float32))
    print(f"Saved positions -> {path}  shape={pos.shape}")


# ---------------------------------------------------------------------------
# CLI: preview synthetic dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trajs, hotspots = generate_dataset()
    print(f"trajs shape: {trajs.shape}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for t in trajs:
        ax.plot(t[:, 0], t[:, 1], alpha=0.5, lw=1)
    for h in hotspots:
        color = "red" if h.kind == "converge" else "blue"
        ax.scatter(h.x, h.y, c=color, s=80, zorder=5, label=h.kind)
    ax.set_title("Synthetic dataset")
    ax.legend()
    plt.tight_layout()
    plt.savefig("synthetic_preview.png", dpi=100)
    print("Saved synthetic_preview.png")
