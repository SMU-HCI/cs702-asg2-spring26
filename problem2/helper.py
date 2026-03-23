"""Helper functions for Problem 2.

- generate_trajectories: Generate clean synthetic trajectories for STL exercises
  (Parts 2.1.1 and 2.1.2).
- export_animation_json: Export animation data as a JSON file for the PyGame viewer.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS, DOT_RADIUS, Hotspot, Hotspot3D


# Colour palette (matches viewer.py).
PALETTE = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (155, 89, 182),
    (243, 156, 18),
    (26, 188, 156),
    (230, 126, 34),
    (52, 73, 94),
    (149, 165, 166),
    (192, 57, 43),
    (41, 128, 185),
    (39, 174, 96),
    (142, 68, 173),
    (211, 84, 0),
    (22, 160, 133),
    (127, 140, 141),
]


# ---------------------------------------------------------------------------
# Trajectory generation for STL exercises (Parts 2.1.1 and 2.1.2)
# ---------------------------------------------------------------------------


def generate_trajectories(
    n: int,
    k: int = 60,
    seed: int | None = None,
    width: int = WINDOW_WIDTH,
    height: int = WINDOW_HEIGHT,
    converge_pos: tuple[float, float] | None = None,
    diverge_pos: tuple[float, float] | None = None,
    t_converge: int | None = None,
    t_diverge: int | None = None,
    noise_std: float = 12.0,
    smooth_sigma: float = 3.0,
    bundle_spread: float = 8.0,
) -> tuple[NDArray, list[Hotspot]]:
    """Generate N clean trajectories that pass through convergence/divergence hotspots.

    Each trajectory interpolates smoothly through four waypoints:
        start -> convergence hotspot -> divergence hotspot -> end

    This produces controlled trajectories suitable for evaluating STL
    specifications in Parts 2.1.1 and 2.1.2.

    Args:
        n:             Number of trajectories.
        k:             Number of time steps.
        seed:          Random seed for reproducibility.
        width:         Canvas width in pixels.
        height:        Canvas height in pixels.
        converge_pos:  (x, y) position of the convergence hotspot.
                       Defaults to (width * 0.4, height * 0.5).
        diverge_pos:   (x, y) position of the divergence hotspot.
                       Defaults to (width * 0.6, height * 0.5).
        t_converge:    Time step for convergence. Defaults to k // 3.
        t_diverge:     Time step for divergence. Defaults to 2 * k // 3.
        noise_std:     Standard deviation of Gaussian noise added to make
                       trajectories wavy. The noise is temporally smoothed.
        smooth_sigma:  Gaussian smoothing sigma (in frames) applied to the
                       noise so motion looks organic rather than jittery.
        bundle_spread: Standard deviation of per-object offsets around each
                       hotspot, so trajectories travel near but don't overlap.

    Returns:
        traj_in:  Shape (N, K, 2) — trajectories (object-major).
        hotspots: List of Hotspot objects.
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    rng = np.random.default_rng(seed)

    # Default hotspot positions and timing.
    if converge_pos is None:
        converge_pos = (width * 0.4, height * 0.5)
    if diverge_pos is None:
        diverge_pos = (width * 0.6, height * 0.5)
    if t_converge is None:
        t_converge = k // 3
    if t_diverge is None:
        t_diverge = 2 * k // 3

    converge_pt = np.array(converge_pos, dtype=np.float32)
    diverge_pt = np.array(diverge_pos, dtype=np.float32)

    # Generate start positions (left side) and end positions (right side).
    # Use stratified sampling so y-positions are spread across the full
    # vertical range rather than clustering in a narrow band.
    margin = height * 0.05
    strata = np.linspace(margin, height - margin, n + 1)

    starts = np.zeros((n, 2), dtype=np.float32)
    starts[:, 0] = rng.uniform(0, width * 0.15, size=n)
    for i in range(n):
        starts[i, 1] = rng.uniform(strata[i], strata[i + 1])
    rng.shuffle(starts, axis=0)

    ends = np.zeros((n, 2), dtype=np.float32)
    ends[:, 0] = rng.uniform(width * 0.85, width, size=n)
    for i in range(n):
        ends[i, 1] = rng.uniform(strata[i], strata[i + 1])
    rng.shuffle(ends, axis=0)

    # Build trajectories via piecewise-linear interpolation through waypoints.
    # Each object gets a small per-object offset around the hotspots so that
    # trajectories travel *near* each other but don't overlap exactly.
    trajs = np.zeros((n, k, 2), dtype=np.float32)

    # Per-object offsets around hotspots.  Use the *same* offset at both
    # hotspots (with a small perturbation) so that the relative spacing
    # between trajectories stays consistent while they travel together.
    offset_base = rng.normal(0, bundle_spread, size=(n, 2)).astype(np.float32)
    offset_c = offset_base
    offset_d = offset_base + rng.normal(0, bundle_spread * 0.2, size=(n, 2)).astype(
        np.float32
    )

    for i in range(n):
        waypoints = np.stack(
            [
                starts[i],
                converge_pt + offset_c[i],
                diverge_pt + offset_d[i],
                ends[i],
            ]
        )  # (4, 2)

        # Allocate frames to each segment proportional to its length,
        # so that the step distance between consecutive points is uniform.
        seg_dists = np.sqrt(np.sum(np.diff(waypoints, axis=0) ** 2, axis=1))  # (3,)
        total_dist = seg_dists.sum()
        # Distribute k-1 intervals across 3 segments proportional to distance.
        raw_alloc = seg_dists / total_dist * (k - 1)
        seg_frames = np.round(raw_alloc).astype(int)
        # Fix rounding so the total is exactly k-1.
        seg_frames[-1] = (k - 1) - seg_frames[:-1].sum()

        idx = 0
        for seg in range(len(waypoints) - 1):
            nf = seg_frames[seg]
            for d in range(2):
                trajs[i, idx : idx + nf, d] = np.linspace(
                    waypoints[seg, d],
                    waypoints[seg + 1, d],
                    nf,
                    endpoint=False,
                )
            idx += nf
        trajs[i, -1] = ends[i]

    # Add Gaussian-smoothed noise to make trajectories wavy (like data.py).
    # The noise is smoothed so motion looks organic rather than jittery.
    from scipy.ndimage import gaussian_filter1d

    noise = rng.normal(0, noise_std, size=trajs.shape).astype(np.float32)
    noise = gaussian_filter1d(noise, sigma=smooth_sigma, axis=1)
    # Zero out start and end so they remain exact.
    noise[:, 0, :] = 0
    noise[:, -1, :] = 0
    trajs += noise
    trajs = np.clip(trajs, 0, [[width, height]])

    # Define hotspots.
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
# 3D trajectory generation (Part 2.3)
# ---------------------------------------------------------------------------

WINDOW_DEPTH: int = 400  # z-axis extent for 3D volume


def generate_trajectories_3d(
    n: int,
    k: int = 60,
    seed: int | None = None,
    width: int = WINDOW_WIDTH,
    height: int = WINDOW_HEIGHT,
    depth: int = WINDOW_DEPTH,
    converge_pos: tuple[float, float, float] | None = None,
    diverge_pos: tuple[float, float, float] | None = None,
    t_converge: int | None = None,
    t_diverge: int | None = None,
    noise_std: float = 12.0,
    smooth_sigma: float = 3.0,
    bundle_spread: float = 8.0,
) -> tuple[NDArray, list[Hotspot3D]]:
    """Generate N clean 3D trajectories through convergence/divergence hotspots.

    Analogous to ``generate_trajectories`` but in a 3D volume of size
    ``width x height x depth`` (default 800 x 600 x 400). Objects start on the
    left face (x ≈ 0), pass through interior hotspots, and exit on the right
    face (x ≈ 800).

    Returns:
        traj_in:  Shape (N, K, 3) — trajectories (object-major).
        hotspots: List of Hotspot3D objects.
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    rng = np.random.default_rng(seed)

    if converge_pos is None:
        converge_pos = (width * 0.4, height * 0.5, depth * 0.5)
    if diverge_pos is None:
        diverge_pos = (width * 0.6, height * 0.5, depth * 0.5)
    if t_converge is None:
        t_converge = k // 3
    if t_diverge is None:
        t_diverge = 2 * k // 3

    converge_pt = np.array(converge_pos, dtype=np.float32)
    diverge_pt = np.array(diverge_pos, dtype=np.float32)

    # Stratified start/end positions on left/right faces.
    margin_y = height * 0.05
    margin_z = depth * 0.05
    strata_y = np.linspace(margin_y, height - margin_y, n + 1)
    strata_z = np.linspace(margin_z, depth - margin_z, n + 1)

    starts = np.zeros((n, 3), dtype=np.float32)
    starts[:, 0] = rng.uniform(0, width * 0.15, size=n)
    for i in range(n):
        starts[i, 1] = rng.uniform(strata_y[i], strata_y[i + 1])
        starts[i, 2] = rng.uniform(strata_z[i], strata_z[i + 1])
    rng.shuffle(starts, axis=0)

    ends = np.zeros((n, 3), dtype=np.float32)
    ends[:, 0] = rng.uniform(width * 0.85, width, size=n)
    for i in range(n):
        ends[i, 1] = rng.uniform(strata_y[i], strata_y[i + 1])
        ends[i, 2] = rng.uniform(strata_z[i], strata_z[i + 1])
    rng.shuffle(ends, axis=0)

    # Build trajectories via piecewise-linear interpolation through waypoints.
    trajs = np.zeros((n, k, 3), dtype=np.float32)

    offset_base = rng.normal(0, bundle_spread, size=(n, 3)).astype(np.float32)
    offset_c = offset_base
    offset_d = offset_base + rng.normal(0, bundle_spread * 0.2, size=(n, 3)).astype(
        np.float32
    )

    for i in range(n):
        waypoints = np.stack(
            [
                starts[i],
                converge_pt + offset_c[i],
                diverge_pt + offset_d[i],
                ends[i],
            ]
        )  # (4, 3)

        seg_dists = np.sqrt(np.sum(np.diff(waypoints, axis=0) ** 2, axis=1))
        total_dist = seg_dists.sum()
        raw_alloc = seg_dists / total_dist * (k - 1)
        seg_frames = np.round(raw_alloc).astype(int)
        seg_frames[-1] = (k - 1) - seg_frames[:-1].sum()

        idx = 0
        for seg in range(len(waypoints) - 1):
            nf = seg_frames[seg]
            for d in range(3):
                trajs[i, idx : idx + nf, d] = np.linspace(
                    waypoints[seg, d],
                    waypoints[seg + 1, d],
                    nf,
                    endpoint=False,
                )
            idx += nf
        trajs[i, -1] = ends[i]

    # Add Gaussian-smoothed noise.
    from scipy.ndimage import gaussian_filter1d

    noise = rng.normal(0, noise_std, size=trajs.shape).astype(np.float32)
    noise = gaussian_filter1d(noise, sigma=smooth_sigma, axis=1)
    noise[:, 0, :] = 0
    noise[:, -1, :] = 0
    trajs += noise
    trajs = np.clip(trajs, 0, [[width, height, depth]])

    hotspots = [
        Hotspot3D(
            x=float(converge_pt[0]),
            y=float(converge_pt[1]),
            z=float(converge_pt[2]),
            kind="converge",
            group=list(range(n)),
            time_step=t_converge,
        ),
        Hotspot3D(
            x=float(diverge_pt[0]),
            y=float(diverge_pt[1]),
            z=float(diverge_pt[2]),
            kind="diverge",
            group=list(range(n)),
            time_step=t_diverge,
        ),
    ]

    return trajs, hotspots


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_animation_json(
    pos: NDArray,
    hotspots: list[Hotspot] | None = None,
    *,
    path: str | Path = "animation.json",
    width: int = WINDOW_WIDTH,
    height: int = WINDOW_HEIGHT,
    fps: int = FPS,
    dot_radius: int = DOT_RADIUS,
    trail_length: int = 10,
    title: str = "Trajectory Animation",
) -> None:
    """Export animation data as a JSON file for the standalone PyGame viewer.

    The JSON contains everything needed to replay the animation:
    positions, hotspot metadata, rendering parameters, and per-object colours.

    Args:
        pos:          Positions array. Accepts either shape (K, N, 2)
                      (time-major) or (N, K, 2) (object-major).  If the
                      array is object-major it is transposed automatically.
        hotspots:     Optional list of Hotspot objects.
        path:         Output JSON file path.
        width:        Canvas width in pixels.
        height:       Canvas height in pixels.
        fps:          Playback frame rate.
        dot_radius:   Dot radius in pixels.
        trail_length: Number of trailing frames to draw.
        title:        Animation title shown in the window.
    """
    hotspots = hotspots or []

    # Auto-detect layout: if axis-0 is smaller assume it's N (object-major).
    if pos.ndim == 3 and pos.shape[0] < pos.shape[1]:
        pos = np.swapaxes(pos, 0, 1)  # (N, K, 2) -> (K, N, 2)

    K, N, _ = pos.shape

    # Per-object colours from the palette.
    colors = [list(PALETTE[i % len(PALETTE)]) for i in range(N)]

    data = {
        "title": title,
        "width": width,
        "height": height,
        "fps": fps,
        "dot_radius": dot_radius,
        "trail_length": trail_length,
        "num_frames": K,
        "num_objects": N,
        "positions": pos.tolist(),
        "colors": colors,
        "hotspots": [
            {
                "x": h.x,
                "y": h.y,
                "kind": h.kind,
                "group": h.group,
                "time_step": h.time_step,
            }
            for h in hotspots
        ],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)

    print(f"Exported animation JSON -> {path}  ({K} frames, {N} objects)")
