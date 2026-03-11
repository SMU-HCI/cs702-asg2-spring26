"""Part 1 — STL Specifications for Animation Qualities.

Write Signal Temporal Logic (STL) specifications that capture desirable
animation properties: bundling, separation, smoothness, and start/end
correctness.  Evaluate each specification against baseline trajectories
with varying numbers of objects.

Usage::

    cd problem2
    python part1.py                        # evaluate on N = 5, 10, 20, 50
    python part1.py --ns 10 20 30          # custom N values
"""

from __future__ import annotations
import argparse
from typing import Any

import numpy as np
import jax.numpy as jnp

from config import K, DOT_RADIUS, RANDOM_SEED, Hotspot

# stljax imports — adjust to match the installed API version.
try:
    import stljax
    from stljax.formula import Predicate, Always, Eventually
    _STLJAX_OK = True
except ImportError:
    _STLJAX_OK = False
    print("[warning] stljax not found; STL robustness will return 0.")


# ═══════════════════════════════════════════════════════════════════════════
# Baseline
# ═══════════════════════════════════════════════════════════════════════════

def baseline(traj_in: np.ndarray) -> np.ndarray:
    """Convert input trajectories to the rendering format without modification.

    Args:
        traj_in: Shape (N, K, 2).

    Returns:
        pos: Shape (K, N, 2).
    """
    return traj_in.transpose(1, 0, 2).copy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# A) Bundling specification
# ═══════════════════════════════════════════════════════════════════════════

def bundling_robustness(
    pos: Any,
    hotspots: list[Hotspot],
    threshold: float = 15.0,
) -> Any:
    """STL robustness for bundling: group members stay close during convergence.

    Args:
        pos:       JAX array, shape (K, N, 2).
        hotspots:  List of Hotspot objects.
        threshold: Maximum allowed pairwise distance within the group (pixels).

    Returns:
        Scalar robustness value (higher = better satisfied).
    """
    # TODO (Part 1 — A): Implement the bundling STL specification.
    #
    #   Suggested approach (differentiable fallback):
    #     For each convergence hotspot:
    #       1. Extract group indices and the convergence time step t_c.
    #       2. Define a time window, e.g., [t_c - 5, t_c + 5].
    #       3. For each frame in the window, compute the maximum pairwise
    #          distance among group members.
    #       4. Robustness = threshold - max over the window of (max pairwise dist).
    #          This corresponds to G_{window}(max_dist < threshold).
    #     Aggregate across hotspots with min (conjunction).
    #
    #   stljax approach:
    #     Define a signal = threshold - max_pairwise_dist_per_frame  (shape (K,))
    #     Use Always(Predicate(...) > 0, interval=[t_lo, t_hi])
    #
    # Placeholder:
    return jnp.array(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# B) Separation (anti-occlusion) specification
# ═══════════════════════════════════════════════════════════════════════════

def separation_robustness(
    pos: Any,
    min_dist: float = 2 * DOT_RADIUS,
) -> Any:
    """STL robustness for separation: all pairs maintain minimum distance at all times.

    Args:
        pos:      JAX array, shape (K, N, 2).
        min_dist: Minimum required separation in pixels.

    Returns:
        Scalar robustness value.
    """
    # TODO (Part 1 — B): Implement the separation STL specification.
    #
    #   Suggested approach:
    #     1. For each frame, compute all pairwise distances: (K, N, N).
    #     2. Extract the minimum pairwise distance per frame (excluding diagonal).
    #     3. Signal = min_pairwise_dist - min_dist   (shape (K,))
    #     4. Robustness of G(signal > 0) = min over all frames of signal.
    #
    #   stljax approach:
    #     signal = min_pairwise_dist_per_frame - min_dist
    #     Use Always(Predicate(...) > 0)
    #
    # Placeholder:
    return jnp.array(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# C) Smoothness specification
# ═══════════════════════════════════════════════════════════════════════════

def smoothness_robustness(
    pos: Any,
    max_accel: float = 5.0,
) -> Any:
    """STL robustness for smoothness: acceleration magnitude stays bounded.    

    Args:
        pos:       JAX array, shape (K, N, 2).
        max_accel: Maximum allowed acceleration magnitude (pixels/frame^2).

    Returns:
        Scalar robustness value.
    """
    # TODO (Part 1 — C): Implement the smoothness STL specification.
    #
    #   Suggested approach:
    #     1. Compute velocity: vel = diff(pos, axis=0)         -> (K-1, N, 2)
    #     2. Compute acceleration: acc = diff(vel, axis=0)     -> (K-2, N, 2)
    #     3. Compute acceleration magnitude per object per frame.
    #     4. Signal = max_accel - max_over_objects(accel_mag)   (shape (K-2,))
    #     5. Robustness of G(signal > 0) = min over frames of signal.
    #
    # Placeholder:
    return jnp.array(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# D) Start/end correctness specification
# ═══════════════════════════════════════════════════════════════════════════

def start_end_robustness(
    pos: Any,
    traj_in: Any,
    tolerance: float = 20.0,
) -> Any:
    """STL robustness for start/end correctness.
        Objects must start and end within a certain distance of their input traj.

    Args:
        pos:       JAX array, shape (K, N, 2).
        traj_in:   JAX array, shape (N, K, 2) — reference trajectories.
        tolerance: Maximum allowed displacement in pixels.

    Returns:
        Scalar robustness value.
    """
    # TODO (Part 1 — D): Implement start/end correctness.
    #
    #   Suggested approach:
    #     start_err = jnp.sqrt(((pos[0] - traj_in[:, 0, :])**2).sum(-1))  # (N,)
    #     end_err   = jnp.sqrt(((pos[-1] - traj_in[:, -1, :])**2).sum(-1))
    #     rho = tolerance - jnp.maximum(start_err, end_err)
    #     return rho.min()   # worst-case over all objects
    #
    # Placeholder:
    return jnp.array(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Combined robustness (used by Part 2)
# ═══════════════════════════════════════════════════════════════════════════

def total_robustness(
    pos: Any,
    traj_in: Any,
    hotspots: list[Hotspot],
) -> Any:
    """Sum all STL robustness terms.

    Args:
        pos:      JAX array, shape (K, N, 2).
        traj_in:  JAX array, shape (N, K, 2).
        hotspots: List of Hotspot objects.

    Returns:
        Scalar total robustness (higher = better).
    """
    rho_bundle = bundling_robustness(pos, hotspots)
    rho_sep = separation_robustness(pos)
    rho_smooth = smoothness_robustness(pos)
    rho_se = start_end_robustness(pos, traj_in)
    return rho_bundle + rho_sep + rho_smooth + rho_se


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation across varying N
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_specifications(ns: list[int] | None = None) -> None:
    """Evaluate all STL specifications on baseline trajectories for varying N.

    For each N, generates a dataset, converts to baseline positions (no
    modification), and reports the robustness of each specification.

    Args:
        ns: List of trajectory counts to evaluate. Defaults to [5, 10, 20, 50].
    """
    from data import generate_dataset

    if ns is None:
        ns = [5, 10, 20, 50]

    print(f"{'N':>5}  {'Bundling':>10}  {'Separation':>11}  {'Smoothness':>11}  {'Start/End':>10}")
    print("-" * 60)

    for n in ns:
        traj_in, hotspots = generate_dataset(n=n, k=K, seed=RANDOM_SEED)
        pos = baseline(traj_in)  # (K, N, 2)

        # Convert to JAX arrays for robustness computation.
        pos_jax = jnp.array(pos, dtype=jnp.float32)
        traj_jax = jnp.array(traj_in, dtype=jnp.float32)

        rho_b = float(bundling_robustness(pos_jax, hotspots))
        rho_s = float(separation_robustness(pos_jax))
        rho_m = float(smoothness_robustness(pos_jax))
        rho_e = float(start_end_robustness(pos_jax, traj_jax))

        print(f"{n:>5}  {rho_b:>10.2f}  {rho_s:>11.2f}  {rho_m:>11.2f}  {rho_e:>10.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Part 1: Evaluate STL specifications on baseline trajectories"
    )
    parser.add_argument(
        "--ns", type=int, nargs="+", default=[5, 10, 20, 50],
        help="List of trajectory counts N to evaluate (default: 5 10 20 50)",
    )
    args = parser.parse_args()

    evaluate_specifications(ns=args.ns)
