"""Part 1 — Animation Algorithm: bundling, layout, and baseline.

Implement a simplified "RouteFlow-like" algorithm that generates an animated
transition from a set of trajectories.

Usage::

    cd problem2_v2
    python part1.py                        # synthetic data
    python part1.py --data path/to.npy     # custom data
    python part1.py --record frames/       # export PNG frames
"""

from __future__ import annotations
import argparse
import numpy as np
from numpy.typing import NDArray

from config import N, K, WINDOW_WIDTH, WINDOW_HEIGHT, DOT_RADIUS, RANDOM_SEED, Hotspot


# ═══════════════════════════════════════════════════════════════════════════
# Baseline
# ═══════════════════════════════════════════════════════════════════════════

def baseline(traj_in: NDArray) -> NDArray:
    """Convert input trajectories to the rendering format without modification.

    Args:
        traj_in: Shape (N, K, 2).

    Returns:
        pos: Shape (K, N, 2).
    """
    return traj_in.transpose(1, 0, 2).copy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Similarity measure
# ═══════════════════════════════════════════════════════════════════════════

def pairwise_similarity(traj_in: NDArray) -> NDArray:
    """Compute an (N, N) pairwise similarity matrix between trajectories.

    Args:
        traj_in: Shape (N, K, 2) — input trajectories.

    Returns:
        Shape (N, N) similarity matrix (higher = more similar).
    """
    N = traj_in.shape[0]
    sim = np.zeros((N, N), dtype=np.float32)

    # TODO (Part 1): Implement a similarity measure.
    #
    #   Option A — Pointwise L2 distance (fast, assumes aligned time):
    #       diff = traj_in[i] - traj_in[j]          # (K, 2)
    #       dist = np.sqrt((diff**2).sum(axis=-1)).mean()
    #       sim[i, j] = 1 / (1 + dist)
    #
    #   Option B — DTW (more robust to time misalignment; use a library
    #       such as dtaidistance or implement a simple O(K^2) version).
    #
    # Placeholder: all similarities set to 1.
    sim[:] = 1.0
    np.fill_diagonal(sim, 0.0)
    return sim


# ═══════════════════════════════════════════════════════════════════════════
# Grouping / clustering
# ═══════════════════════════════════════════════════════════════════════════

def cluster_trajectories(
    traj_in: NDArray,
    n_clusters: int = 4,
    sim_matrix: NDArray | None = None,
) -> list[list[int]]:
    """Assign each trajectory to a group (cluster).

    Args:
        traj_in:    Shape (N, K, 2).
        n_clusters: Number of groups to form.
        sim_matrix: Optional precomputed (N, N) similarity matrix.

    Returns:
        List of groups; each group is a list of trajectory indices.
    """
    N = traj_in.shape[0]
    groups: list[list[int]] = [[] for _ in range(n_clusters)]

    # TODO (Part 1): Implement trajectory clustering.
    #
    #   Suggested approach using sklearn:
    #       from sklearn.cluster import AgglomerativeClustering
    #       flat = traj_in.reshape(N, -1)
    #       labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(flat)
    #
    #   Or use the similarity matrix with spectral clustering:
    #       from sklearn.cluster import SpectralClustering
    #       labels = SpectralClustering(n_clusters, affinity='precomputed').fit_predict(sim_matrix)
    #
    # Placeholder: round-robin assignment.
    for i in range(N):
        groups[i % n_clusters].append(i)

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# Bundling
# ═══════════════════════════════════════════════════════════════════════════

def bundle_trajectories(
    traj_in: NDArray,
    hotspots: list[Hotspot] | None = None,
    n_clusters: int = 4,
    bundle_strength: float = 0.5,
    smooth_sigma: float = 3.0,
) -> tuple[NDArray, list[list[int]]]:
    """Bundle trajectories toward group centrelines.

    Args:
        traj_in:        Shape (N, K, 2) — input trajectories.
        hotspots:       Known hotspot locations (used to anchor bundling).
        n_clusters:     Number of groups.
        bundle_strength: How strongly to attract objects to the centroid [0,1].
        smooth_sigma:   Gaussian smoothing applied to centrelines.

    Returns:
        bundled: Shape (N, K, 2) — bundled trajectories.
        groups:  List of groups (lists of indices).
    """
    N, K, D = traj_in.shape
    bundled = traj_in.copy()

    # TODO (Part 1): Implement the bundling algorithm.
    #
    #   Suggested force-based approach:
    #     1. Cluster trajectories -> groups.
    #     2. For each group, compute the per-timestep centroid (centreline).
    #        Optionally smooth the centreline with Gaussian smoothing.
    #     3. Move each trajectory toward its group centreline by bundle_strength:
    #            bundled[i] = (1 - bundle_strength) * traj_in[i]
    #                          + bundle_strength * centreline[group[i]]
    #     4. If hotspots are provided, keep bundled trajectories close to
    #        the hotspot locations at the specified time steps (anchor).
    #
    # Placeholder: return input unchanged.
    groups = cluster_trajectories(traj_in, n_clusters)
    return bundled, groups


# ═══════════════════════════════════════════════════════════════════════════
# Layout / anti-occlusion ("seat allocation")
# ═══════════════════════════════════════════════════════════════════════════

def assign_offsets(
    bundled: NDArray,
    groups: list[list[int]],
    dot_radius: float = DOT_RADIUS,
    method: str = "static",
) -> NDArray:
    """Assign per-object layout offsets to reduce occlusion within groups.

    Args:
        bundled:    Shape (N, K, 2) — bundled trajectories from Part 1.
        groups:     List of groups (lists of object indices).
        dot_radius: Visual dot radius (used to set minimum spacing).
        method:     ``"static"`` (perpendicular lanes) or ``"repulsion"``.

    Returns:
        pos: Shape (K, N, 2) — final positions ready for rendering.
             Note the axis order change to (K, N, 2).
    """
    N, K, D = bundled.shape
    pos = bundled.copy()  # (N, K, 2)

    # TODO (Part 1): Implement the layout strategy.
    #
    #   Option A — Static perpendicular offsets:
    #     For each group at each time step:
    #       1. Compute the group centroid.
    #       2. Compute the direction of motion (tangent to the centreline).
    #       3. Compute the perpendicular direction.
    #       4. Space objects along the perpendicular: offsets like
    #            [-spacing*(n_g-1)/2, ..., +spacing*(n_g-1)/2]
    #          where spacing = 2 * dot_radius + margin.
    #
    #   Option B — Repulsion forces:
    #     Iterate for a fixed number of steps:
    #       For each pair (i, j) in the same group:
    #         if dist(pos[i,t], pos[j,t]) < 2 * dot_radius:
    #             apply a repulsive force to push them apart.
    #     Add a damping / spring-back force toward the centreline.
    #
    # Placeholder: no offset (returns bundled positions in time-major order).

    # Convert from (N, K, 2) -> (K, N, 2).
    pos_out = pos.transpose(1, 0, 2).copy()
    return pos_out


# ═══════════════════════════════════════════════════════════════════════════
# Run Part 1 pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run(
    data_path: str | None = None,
    n_clusters: int = 4,
    bundle_strength: float = 0.5,
    record_dir: str | None = None,
) -> None:
    from utils import set_global_seed
    from data import generate_dataset, load_trajectories
    import metrics

    set_global_seed(RANDOM_SEED)

    # --- Load / generate data ---
    if data_path is None:
        print("Generating synthetic dataset ...")
        traj_in, hotspots = generate_dataset(n=N, k=K)
    else:
        print(f"Loading trajectories from {data_path} ...")
        traj_in = load_trajectories(data_path, resample_to=K)
        hotspots = []

    print(f"traj_in shape: {traj_in.shape}")  # (N, K, 2)

    # --- Baseline (no processing) ---
    pos_baseline = baseline(traj_in)   # (K, N, 2)

    # --- Part 1 algorithm ---
    bundled, groups = bundle_trajectories(
        traj_in,
        hotspots=hotspots,
        n_clusters=n_clusters,
        bundle_strength=bundle_strength,
    )
    pos_part1 = assign_offsets(bundled, groups)  # (K, N, 2)

    # --- Evaluate ---
    print("\n--- Baseline metrics ---")
    print(f"  occlusion       : {metrics.occlusion_rate(pos_baseline):.4f}")
    print(f"  mean deviation  : {metrics.mean_deviation(traj_in, pos_baseline):.2f} px")
    sm = metrics.smoothness_score(pos_baseline)
    print(f"  mean velocity   : {sm['mean_velocity']:.2f}")
    print(f"  mean accel      : {sm['mean_acceleration']:.2f}")

    print("\n--- Part 1 metrics ---")
    print(f"  occlusion       : {metrics.occlusion_rate(pos_part1):.4f}")
    print(f"  mean deviation  : {metrics.mean_deviation(traj_in, pos_part1):.2f} px")
    sm1 = metrics.smoothness_score(pos_part1)
    print(f"  mean velocity   : {sm1['mean_velocity']:.2f}")
    print(f"  mean accel      : {sm1['mean_acceleration']:.2f}")

    # --- Render / record ---
    if record_dir is not None:
        from viewer import export_frames
        export_frames(pos_part1, record_dir)
    else:
        from viewer import play
        play(pos_part1, title="Part 1 — Bundled Animation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Part 1 animation pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to .npy trajectory file")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--record", type=str, default=None, help="Export frames to this directory")
    args = parser.parse_args()

    run(
        data_path=args.data,
        n_clusters=args.clusters,
        bundle_strength=args.strength,
        record_dir=args.record,
    )
