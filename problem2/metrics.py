"""Evaluation metrics: occlusion, deviation, smoothness, and dispersion."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from config import DOT_RADIUS


# ---------------------------------------------------------------------------
# Occlusion
# ---------------------------------------------------------------------------

def occlusion_rate(pos: NDArray, radius: float = DOT_RADIUS) -> float:
    """Compute the fraction of (frame, pair) combinations where two dots overlap.

    Two dots are considered occluded when the distance between their centres
    is less than ``2 * radius``.

    Args:
        pos:    Shape (K, N, 2) — output positions.
        radius: Visual radius of each dot in pixels.

    Returns:
        Occlusion rate in [0, 1].
    """
    K, N, _ = pos.shape
    threshold = 2.0 * radius
    n_pairs = N * (N - 1) // 2
    if n_pairs == 0:
        return 0.0

    occluded = 0
    total = K * n_pairs

    for t in range(K):
        pts = pos[t]  # (N, 2)
        diff = pts[:, None, :] - pts[None, :, :]   # (N, N, 2)
        dist = np.sqrt((diff ** 2).sum(axis=-1))    # (N, N)
        # Upper triangle only (avoid double counting).
        mask = np.triu(dist < threshold, k=1)
        occluded += mask.sum()

    return occluded / total


def within_group_occlusion(pos: NDArray, groups: list[list[int]], radius: float = DOT_RADIUS) -> float:
    """Occlusion rate restricted to pairs within the same group.

    Args:
        pos:    Shape (K, N, 2).
        groups: List of groups; each group is a list of object indices.
        radius: Dot radius.

    Returns:
        Within-group occlusion rate in [0, 1].
    """
    K = pos.shape[0]
    threshold = 2.0 * radius
    occluded = 0
    total = 0

    for group in groups:
        g = list(group)
        n_g = len(g)
        if n_g < 2:
            continue
        n_pairs = n_g * (n_g - 1) // 2
        total += K * n_pairs
        for t in range(K):
            pts = pos[t][g]  # (n_g, 2)
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=-1))
            mask = np.triu(dist < threshold, k=1)
            occluded += mask.sum()

    if total == 0:
        return 0.0
    return occluded / total


# ---------------------------------------------------------------------------
# Deviation
# ---------------------------------------------------------------------------

def mean_deviation(traj_in: NDArray, pos: NDArray) -> float:
    """Mean per-object per-frame L2 deviation from the input trajectories.

    Args:
        traj_in: Shape (N, K, 2) — input (reference) trajectories.
        pos:     Shape (K, N, 2) — output positions.

    Returns:
        Mean deviation in pixels.
    """
    pos_nm = pos.transpose(1, 0, 2)  # -> (N, K, 2)
    diff = pos_nm - traj_in
    return float(np.sqrt((diff ** 2).sum(axis=-1)).mean())


def max_deviation(traj_in: NDArray, pos: NDArray) -> float:
    """Maximum per-object per-frame L2 deviation from the input trajectories.

    Args:
        traj_in: Shape (N, K, 2).
        pos:     Shape (K, N, 2).

    Returns:
        Maximum deviation in pixels.
    """
    pos_nm = pos.transpose(1, 0, 2)  # -> (N, K, 2)
    diff = pos_nm - traj_in
    return float(np.sqrt((diff ** 2).sum(axis=-1)).max())


# ---------------------------------------------------------------------------
# Smoothness
# ---------------------------------------------------------------------------

def mean_velocity(pos: NDArray, dt: float = 1.0) -> float:
    """Mean L2 velocity magnitude across all objects and frames.

    Args:
        pos: Shape (K, N, 2).
        dt:  Time step (seconds or arbitrary units).

    Returns:
        Mean speed in pixels/dt.
    """
    vel = np.diff(pos, axis=0) / dt          # (K-1, N, 2)
    speed = np.sqrt((vel ** 2).sum(axis=-1)) # (K-1, N)
    return float(speed.mean())


def mean_acceleration(pos: NDArray, dt: float = 1.0) -> float:
    """Mean L2 acceleration magnitude across all objects and frames.

    Args:
        pos: Shape (K, N, 2).
        dt:  Time step.

    Returns:
        Mean acceleration in pixels/dt^2.
    """
    vel = np.diff(pos, axis=0) / dt          # (K-1, N, 2)
    acc = np.diff(vel, axis=0) / dt          # (K-2, N, 2)
    acc_mag = np.sqrt((acc ** 2).sum(axis=-1))
    return float(acc_mag.mean())


def smoothness_score(pos: NDArray, dt: float = 1.0) -> dict[str, float]:
    """Return a dictionary with both velocity and acceleration metrics.

    Args:
        pos: Shape (K, N, 2).
        dt:  Time step.

    Returns:
        Dict with keys ``mean_velocity`` and ``mean_acceleration``.
    """
    return {
        "mean_velocity": mean_velocity(pos, dt),
        "mean_acceleration": mean_acceleration(pos, dt),
    }


# ---------------------------------------------------------------------------
# Dispersion / compactness
# ---------------------------------------------------------------------------

def mean_dispersion(pos: NDArray) -> float:
    """Mean per-frame standard deviation of positions (spread of the swarm).

    Args:
        pos: Shape (K, N, 2).

    Returns:
        Mean dispersion in pixels (average std across frames and dimensions).
    """
    std = pos.std(axis=1)   # (K, 2)
    return float(std.mean())


def group_compactness(pos: NDArray, groups: list[list[int]]) -> dict[int, float]:
    """Mean within-group spread over time, per group.

    Args:
        pos:    Shape (K, N, 2).
        groups: List of groups (lists of object indices).

    Returns:
        Dict mapping group index -> mean intra-group std (pixels).
    """
    result: dict[int, float] = {}
    for g_idx, group in enumerate(groups):
        if len(group) < 2:
            result[g_idx] = 0.0
            continue
        group_pos = pos[:, group, :]   # (K, n_g, 2)
        std = group_pos.std(axis=1)    # (K, 2)
        result[g_idx] = float(std.mean())
    return result
