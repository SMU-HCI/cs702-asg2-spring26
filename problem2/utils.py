"""Utility functions: geometry, resampling, and reproducibility."""

from __future__ import annotations
import random
import numpy as np
from numpy.typing import NDArray

try:
    import jax
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and JAX (if available)."""
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int):
    """Return a JAX PRNGKey if JAX is available, otherwise None."""
    if _JAX_AVAILABLE:
        return jax.random.PRNGKey(seed)
    return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pairwise_distances(pts: NDArray) -> NDArray:
    """Return the (N, N) matrix of Euclidean distances between N 2-D points.

    Args:
        pts: Shape (N, 2).

    Returns:
        Shape (N, N) distance matrix.
    """
    diff = pts[:, None, :] - pts[None, :, :]   # (N, N, 2)
    return np.sqrt((diff ** 2).sum(axis=-1))


def interpolate_polyline(pts: NDArray, t: float) -> NDArray:
    """Linearly interpolate along a polyline at normalised parameter t in [0,1].

    Args:
        pts: Shape (M, 2) — polyline vertices.
        t:   Parameter in [0, 1].

    Returns:
        Shape (2,) interpolated point.
    """
    if len(pts) == 1:
        return pts[0]
    t = float(np.clip(t, 0.0, 1.0))
    idx_f = t * (len(pts) - 1)
    i = int(idx_f)
    frac = idx_f - i
    if i >= len(pts) - 1:
        return pts[-1]
    return (1 - frac) * pts[i] + frac * pts[i + 1]


def smooth_polyline(pts: NDArray, sigma: float = 2.0) -> NDArray:
    """Apply Gaussian smoothing to a polyline.

    Args:
        pts:   Shape (M, 2).
        sigma: Smoothing standard deviation in samples.

    Returns:
        Smoothed polyline of the same shape.
    """
    from scipy.ndimage import gaussian_filter1d
    return np.stack(
        [gaussian_filter1d(pts[:, d], sigma=sigma) for d in range(pts.shape[1])],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample(traj: NDArray, K: int) -> NDArray:
    """Resample a single trajectory to exactly K time steps.

    Arc-length parameterisation is used so that points are evenly spaced
    along the path rather than in time.

    Args:
        traj: Shape (T, 2) — original trajectory with T (possibly uneven) steps.
        K:    Target number of time steps.

    Returns:
        Shape (K, 2) resampled trajectory.
    """
    T = len(traj)
    if T == K:
        return traj.copy()

    # Compute cumulative arc length.
    diffs = np.diff(traj, axis=0)                         # (T-1, 2)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))       # (T-1,)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)]) # (T,)

    if arc[-1] == 0.0:
        # Degenerate: all points are the same.
        return np.tile(traj[0], (K, 1))

    # Uniform query points along arc length.
    arc_query = np.linspace(0.0, arc[-1], K)

    # Interpolate each dimension separately.
    resampled = np.stack(
        [np.interp(arc_query, arc, traj[:, d]) for d in range(traj.shape[1])],
        axis=1,
    )
    return resampled.astype(traj.dtype)


def resample_all(trajs: NDArray, K: int) -> NDArray:
    """Resample a batch of trajectories to K steps.

    Args:
        trajs: Shape (N, T, 2).
        K:     Target number of time steps.

    Returns:
        Shape (N, K, 2).
    """
    return np.stack([resample(t, K) for t in trajs], axis=0)
