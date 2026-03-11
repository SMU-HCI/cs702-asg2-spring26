"""Part 2.3 — 3D Trajectory Optimization with Rerun Visualization.

Extend the STL-based robustness calculation and trajectory optimization
from Parts 2.1/2.2 to three-dimensional space, and visualize results
using the Rerun SDK.

Usage::

    cd problem2
    python part3.py                        # optimize and visualize
    python part3.py --steps 500 --lr 0.005
    python part3.py --baseline-only        # visualize baseline without optimization
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import optax
import rerun as rr

from config import N, K, RANDOM_SEED, N_STEPS, LEARNING_RATE, LOG_EVERY
from config import W_STL, W_SMOOTH, W_DEVIATION, W_SEPARATION

# stljax imports (optional, same as Part 1).
try:
    import stljax
    _STLJAX_OK = True
except ImportError:
    _STLJAX_OK = False
    print("[warning] stljax not found; STL robustness will return 0.")


# ═══════════════════════════════════════════════════════════════════════════
# 3D Hotspot type
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Hotspot3D:
    x: float
    y: float
    z: float
    kind: str          # "converge" | "diverge"
    group: list[int]   # indices of objects belonging to this hotspot
    time_step: int     # frame at which the hotspot is active


# ═══════════════════════════════════════════════════════════════════════════
# 3D Dataset generation
# ═══════════════════════════════════════════════════════════════════════════

VOLUME_W, VOLUME_H, VOLUME_D = 800, 600, 400  # 3D bounding volume


def generate_dataset_3d(
    n: int = N,
    k: int = K,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, list[Hotspot3D]]:
    """Generate N synthetic 3D trajectories of length K with two hotspots.

    Objects start on the left face (x ≈ 0), pass through a convergence
    hotspot and a divergence hotspot in the interior, and exit on the
    right face (x ≈ VOLUME_W).

    Args:
        n:    Number of trajectories.
        k:    Number of time steps.
        seed: Random seed.

    Returns:
        traj_in:  Shape (N, K, 3) — input 3D trajectories.
        hotspots: List of Hotspot3D objects.
    """
    rng = np.random.default_rng(seed)

    # Hotspot positions in the interior.
    converge_pt = np.array(
        [VOLUME_W * 0.4, VOLUME_H * 0.5, VOLUME_D * 0.5], dtype=np.float32
    )
    diverge_pt = np.array(
        [VOLUME_W * 0.6, VOLUME_H * 0.5, VOLUME_D * 0.5], dtype=np.float32
    )

    t_converge = k // 3
    t_diverge = 2 * k // 3

    # Start/end positions on left/right faces.
    starts = np.column_stack([
        rng.uniform(0, VOLUME_W * 0.15, size=n),
        rng.uniform(0, VOLUME_H, size=n),
        rng.uniform(0, VOLUME_D, size=n),
    ]).astype(np.float32)

    ends = np.column_stack([
        rng.uniform(VOLUME_W * 0.85, VOLUME_W, size=n),
        rng.uniform(0, VOLUME_H, size=n),
        rng.uniform(0, VOLUME_D, size=n),
    ]).astype(np.float32)

    # Per-object offsets around hotspots (messy real-world data).
    offset_c = rng.normal(0, 60, size=(n, 3)).astype(np.float32)
    offset_d = rng.normal(0, 60, size=(n, 3)).astype(np.float32)
    converge_pts = converge_pt + offset_c
    diverge_pts = diverge_pt + offset_d

    # Build trajectories: start -> converge -> diverge -> end.
    trajs = np.zeros((n, k, 3), dtype=np.float32)
    waypoints_t = [0, t_converge, t_diverge, k - 1]

    for i in range(n):
        wps = np.stack([starts[i], converge_pts[i], diverge_pts[i], ends[i]])
        for seg in range(len(waypoints_t) - 1):
            t0, t1 = waypoints_t[seg], waypoints_t[seg + 1]
            steps = t1 - t0
            for d in range(3):
                trajs[i, t0:t1, d] = np.linspace(wps[seg, d], wps[seg + 1, d], steps)
        trajs[i, -1] = ends[i]

    # Add smoothed noise.
    from scipy.ndimage import gaussian_filter1d
    noise = rng.normal(0, 12, size=trajs.shape).astype(np.float32)
    noise = gaussian_filter1d(noise, sigma=1.5, axis=1)
    trajs += noise
    trajs = np.clip(trajs, 0, [[VOLUME_W, VOLUME_H, VOLUME_D]])

    hotspots = [
        Hotspot3D(
            x=float(converge_pt[0]), y=float(converge_pt[1]),
            z=float(converge_pt[2]),
            kind="converge", group=list(range(n)), time_step=t_converge,
        ),
        Hotspot3D(
            x=float(diverge_pt[0]), y=float(diverge_pt[1]),
            z=float(diverge_pt[2]),
            kind="diverge", group=list(range(n)), time_step=t_diverge,
        ),
    ]

    return trajs, hotspots


# ═══════════════════════════════════════════════════════════════════════════
# Baseline
# ═══════════════════════════════════════════════════════════════════════════

def baseline_3d(traj_in: np.ndarray) -> np.ndarray:
    """Convert (N, K, 3) input trajectories to (K, N, 3) without modification."""
    return traj_in.transpose(1, 0, 2).copy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# A) 3D STL Specifications
# ═══════════════════════════════════════════════════════════════════════════

def bundling_robustness_3d(
    pos: Any,
    hotspots: list[Hotspot3D],
    threshold: float = 20.0,
) -> Any:
    """STL robustness for bundling in 3D: group members stay close during convergence.

    Args:
        pos:       JAX array, shape (K, N, 3).
        hotspots:  List of Hotspot3D objects.
        threshold: Maximum allowed pairwise distance within the group (pixels).

    Returns:
        Scalar robustness value.
    """
    # TODO (Part 2.3 — A): Implement 3D bundling STL specification.
    #
    #   Same approach as 2D but with 3D distances:
    #     For each convergence hotspot:
    #       1. Extract group indices and convergence time step t_c.
    #       2. Define a time window [t_c - 5, t_c + 5].
    #       3. For each frame in the window, compute max pairwise distance
    #          among group members in R^3.
    #       4. Robustness = threshold - max over window of (max pairwise dist).
    #     Aggregate across hotspots with min (conjunction).
    #
    # Placeholder:
    return jnp.array(0.0)


def separation_robustness_3d(
    pos: Any,
    min_dist: float = 10.0,
) -> Any:
    """STL robustness for separation in 3D: all pairs maintain minimum distance.

    Args:
        pos:      JAX array, shape (K, N, 3).
        min_dist: Minimum required separation in pixels.

    Returns:
        Scalar robustness value.
    """
    # TODO (Part 2.3 — A): Implement 3D separation STL specification.
    #
    #   Same approach as 2D:
    #     1. For each frame, compute all pairwise distances in R^3: (K, N, N).
    #     2. Extract min pairwise distance per frame (excluding diagonal).
    #     3. Signal = min_pairwise_dist - min_dist   (shape (K,))
    #     4. Robustness of G(signal > 0) = min over all frames.
    #
    # Placeholder:
    return jnp.array(0.0)


def smoothness_robustness_3d(
    pos: Any,
    max_accel: float = 5.0,
) -> Any:
    """STL robustness for smoothness in 3D: acceleration magnitude stays bounded.

    Args:
        pos:       JAX array, shape (K, N, 3).
        max_accel: Maximum allowed acceleration magnitude.

    Returns:
        Scalar robustness value.
    """
    # TODO (Part 2.3 — A): Implement 3D smoothness STL specification.
    #
    #   1. vel = diff(pos, axis=0)       -> (K-1, N, 3)
    #   2. acc = diff(vel, axis=0)       -> (K-2, N, 3)
    #   3. acc_mag = norm(acc, axis=-1)  -> (K-2, N)
    #   4. signal = max_accel - max over objects of acc_mag  -> (K-2,)
    #   5. Robustness = min over frames of signal.
    #
    # Placeholder:
    return jnp.array(0.0)


def start_end_robustness_3d(
    pos: Any,
    traj_in: Any,
    tolerance: float = 25.0,
) -> Any:
    """STL robustness for start/end correctness in 3D.

    Args:
        pos:       JAX array, shape (K, N, 3).
        traj_in:   JAX array, shape (N, K, 3) — reference trajectories.
        tolerance: Maximum allowed displacement in pixels.

    Returns:
        Scalar robustness value.
    """
    # TODO (Part 2.3 - A): Implement 3D start/end correctness.
    #
    #   start_err = norm(pos[0] - traj_in[:, 0, :], axis=-1)   -> (N,)
    #   end_err   = norm(pos[-1] - traj_in[:, -1, :], axis=-1) -> (N,)
    #   rho = tolerance - max(start_err, end_err)
    #   return rho.min()
    #
    # Placeholder:
    return jnp.array(0.0)


def total_robustness_3d(
    pos: Any,
    traj_in: Any,
    hotspots: list[Hotspot3D],
) -> Any:
    """Sum all 3D STL robustness terms."""
    rho_bundle = bundling_robustness_3d(pos, hotspots)
    rho_sep = separation_robustness_3d(pos)
    rho_smooth = smoothness_robustness_3d(pos)
    rho_se = start_end_robustness_3d(pos, traj_in)
    return rho_bundle + rho_sep + rho_smooth + rho_se


# ═══════════════════════════════════════════════════════════════════════════
# B) 3D Differentiable Losses
# ═══════════════════════════════════════════════════════════════════════════

def smoothness_loss_3d(pos: Any) -> Any:
    """Velocity + acceleration penalty in 3D.

    Args:
        pos: JAX array, shape (K, N, 3).

    Returns:
        Scalar penalty.
    """
    vel = jnp.diff(pos, axis=0)
    acc = jnp.diff(vel, axis=0)
    return (vel ** 2).mean() + (acc ** 2).mean()


def deviation_loss_3d(pos: Any, traj_in: Any) -> Any:
    """Mean squared deviation from input trajectories in 3D.

    Args:
        pos:     JAX array, shape (K, N, 3).
        traj_in: JAX array, shape (N, K, 3).

    Returns:
        Scalar penalty.
    """
    ref = jnp.transpose(traj_in, (1, 0, 2))  # (K, N, 3)
    return ((pos - ref) ** 2).mean()


def separation_loss_3d(pos: Any, min_dist: float = 10.0) -> Any:
    """Penalise pairs of objects closer than min_dist in 3D.

    Args:
        pos:      JAX array, shape (K, N, 3).
        min_dist: Minimum desired separation.

    Returns:
        Scalar penalty.
    """
    # TODO (Part 2.3 — B): Implement 3D separation penalty.
    #
    #   diff = pos[:, :, None, :] - pos[:, None, :, :]   # (K, N, N, 3)
    #   dist2 = (diff**2).sum(-1)                         # (K, N, N)
    #   penalty = jnp.maximum(0.0, min_dist**2 - dist2)
    #   mask = 1 - jnp.eye(pos.shape[1])
    #   return (penalty * mask).mean()
    #
    # Placeholder:
    return jnp.array(0.0)


def total_loss_3d(
    pos: Any,
    traj_in: Any,
    hotspots: list[Hotspot3D],
    w_stl: float = W_STL,
    w_smooth: float = W_SMOOTH,
    w_deviation: float = W_DEVIATION,
    w_separation: float = W_SEPARATION,
) -> Any:
    """Combined 3D optimization objective (minimise)."""
    stl_rho = total_robustness_3d(pos, traj_in, hotspots)
    l_smooth = smoothness_loss_3d(pos)
    l_deviation = deviation_loss_3d(pos, traj_in)
    l_sep = separation_loss_3d(pos)

    loss = (
        -w_stl * stl_rho
        + w_smooth * l_smooth
        + w_deviation * l_deviation
        + w_separation * l_sep
    )
    return loss


# ═══════════════════════════════════════════════════════════════════════════
# B) 3D Optimization loop
# ═══════════════════════════════════════════════════════════════════════════

def optimize_3d(
    traj_in: np.ndarray,
    hotspots: list[Hotspot3D] | None = None,
    n_steps: int = N_STEPS,
    lr: float = LEARNING_RATE,
    log_every: int = LOG_EVERY,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Optimise 3D trajectory positions via gradient descent.

    Args:
        traj_in:   Shape (N, K, 3) — input reference trajectories.
        hotspots:  Hotspot3D objects for STL specs.
        n_steps:   Number of gradient steps.
        lr:        Adam learning rate.
        log_every: Print diagnostics every this many steps.

    Returns:
        pos_opt:  Shape (K, N, 3) optimised positions (NumPy, float32).
        history:  Dict with keys "loss" and "step".
    """
    hotspots = hotspots or []

    init_pos = traj_in.transpose(1, 0, 2).copy()  # (K, N, 3)

    traj_jax = jnp.array(traj_in, dtype=jnp.float32)
    pos = jnp.array(init_pos, dtype=jnp.float32)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(pos)

    @jax.jit
    def step(pos, opt_state):
        loss_val, grads = jax.value_and_grad(total_loss_3d)(
            pos, traj_jax, hotspots
        )
        updates, opt_state_new = optimizer.update(grads, opt_state)
        pos_new = optax.apply_updates(pos, updates)
        return pos_new, opt_state_new, loss_val

    history: dict[str, list[float]] = {"loss": [], "step": []}

    print(f"Starting 3D optimisation: {n_steps} steps, lr={lr}")
    for i in range(n_steps):
        pos, opt_state, loss_val = step(pos, opt_state)

        if i % log_every == 0 or i == n_steps - 1:
            lv = float(loss_val)
            history["loss"].append(lv)
            history["step"].append(i)
            print(f"  step {i:5d} / {n_steps}   loss = {lv:.4f}")

    pos_opt = np.array(pos, dtype=np.float32)
    return pos_opt, history


# ═══════════════════════════════════════════════════════════════════════════
# C) Rerun Visualization
# ═══════════════════════════════════════════════════════════════════════════

# Color palette for objects (RGB, 0–255).
PALETTE_3D = [
    (230, 25, 75),   (60, 180, 75),   (255, 225, 25),  (0, 130, 200),
    (245, 130, 48),  (145, 30, 180),  (70, 240, 240),  (240, 50, 230),
    (210, 245, 60),  (250, 190, 212), (0, 128, 128),   (220, 190, 255),
    (170, 110, 40),  (255, 250, 200), (128, 0, 0),     (170, 255, 195),
    (128, 128, 0),   (255, 215, 180), (0, 0, 128),     (128, 128, 128),
]


def visualize_rerun(
    pos: np.ndarray,
    hotspots: list[Hotspot3D],
    traj_in: np.ndarray | None = None,
    title: str = "3D Trajectory Optimization",
) -> None:
    """Visualize 3D trajectories using the Rerun SDK.

    Args:
        pos:      Shape (K, N, 3) — positions to visualize.
        hotspots: Hotspot3D objects to display as markers.
        traj_in:  Shape (N, K, 3) — original input trajectories (optional,
                  for comparison).
        title:    Title for the Rerun recording.
    """
    K_frames, N_objs, _ = pos.shape

    rr.init(title, spawn=True)

    colors = [PALETTE_3D[i % len(PALETTE_3D)] for i in range(N_objs)]

    # 1. Log full trajectory lines (static, outside any timeline).
    for i in range(N_objs):
        rr.log(
            f"world/trajectories/obj_{i}",
            rr.LineStrips3D(
                [pos[:, i, :].tolist()],
                colors=[colors[i]],
            ),
        )

    # 2. Log baseline trajectories as semi-transparent lines for comparison.
    if traj_in is not None:
        for i in range(N_objs):
            r, g, b = colors[i]
            rr.log(
                f"world/baseline/obj_{i}",
                rr.LineStrips3D(
                    [traj_in[i, :, :].tolist()],
                    colors=[(r, g, b, 60)],
                ),
            )

    # 3. Log hotspot markers (static).
    converge_pts = []
    converge_colors = []
    diverge_pts = []
    diverge_colors = []
    for h in hotspots:
        if h.kind == "converge":
            converge_pts.append([h.x, h.y, h.z])
            converge_colors.append((220, 60, 60))
        else:
            diverge_pts.append([h.x, h.y, h.z])
            diverge_colors.append((60, 60, 220))

    if converge_pts:
        rr.log(
            "world/hotspots/converge",
            rr.Points3D(converge_pts, colors=converge_colors, radii=[15.0]),
        )
    if diverge_pts:
        rr.log(
            "world/hotspots/diverge",
            rr.Points3D(diverge_pts, colors=diverge_colors, radii=[15.0]),
        )

    # 4. Animate object positions over time.
    for t in range(K_frames):
        rr.set_time("frame", sequence=t)
        positions = pos[t]  # (N, 3)
        rr.log(
            "world/objects",
            rr.Points3D(positions, colors=colors, radii=[5.0]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_3d(traj_in: np.ndarray, pos: np.ndarray, hotspots: list[Hotspot3D]) -> dict:
    """Compute all 3D STL robustness values for a set of positions.

    Args:
        traj_in:  Shape (N, K, 3).
        pos:      Shape (K, N, 3).
        hotspots: Hotspot3D list.

    Returns:
        Dict with robustness values for each specification.
    """
    pos_jax = jnp.array(pos, dtype=jnp.float32)
    traj_jax = jnp.array(traj_in, dtype=jnp.float32)

    return {
        "bundling": float(bundling_robustness_3d(pos_jax, hotspots)),
        "separation": float(separation_robustness_3d(pos_jax)),
        "smoothness": float(smoothness_robustness_3d(pos_jax)),
        "start_end": float(start_end_robustness_3d(pos_jax, traj_jax)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main(
    n_steps: int = N_STEPS,
    lr: float = LEARNING_RATE,
    baseline_only: bool = False,
) -> None:
    print("Generating 3D dataset ...")
    traj_in, hotspots = generate_dataset_3d(n=N, k=K)
    print(f"traj_in shape: {traj_in.shape}")

    # Baseline.
    pos_baseline = baseline_3d(traj_in)

    print("\n--- Baseline robustness ---")
    baseline_eval = evaluate_3d(traj_in, pos_baseline, hotspots)
    for k_name, v in baseline_eval.items():
        print(f"  {k_name:>12s}: {v:.2f}")

    if baseline_only:
        visualize_rerun(pos_baseline, hotspots, traj_in=traj_in,
                        title="3D Baseline Trajectories")
        return

    # Optimize.
    pos_opt, history = optimize_3d(traj_in, hotspots=hotspots,
                                   n_steps=n_steps, lr=lr)

    print("\n--- Optimised robustness ---")
    opt_eval = evaluate_3d(traj_in, pos_opt, hotspots)
    for k_name, v in opt_eval.items():
        print(f"  {k_name:>12s}: {v:.2f}")

    # Visualize.
    visualize_rerun(pos_opt, hotspots, traj_in=traj_in,
                    title="3D Optimised Trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Part 2.3: 3D Trajectory Optimization with Rerun"
    )
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help="Number of optimisation steps")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Adam learning rate")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Visualize baseline only (skip optimization)")
    args = parser.parse_args()

    main(n_steps=args.steps, lr=args.lr, baseline_only=args.baseline_only)
