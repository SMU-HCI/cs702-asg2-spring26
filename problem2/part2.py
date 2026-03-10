"""Part 2 — STL + Gradient Programming: specs, losses, and optimisation.

Express animation intent as Signal Temporal Logic (STL) formulas, then
optimise trajectories via JAX gradients.

Usage::

    cd problem2_v2
    python part2.py                        # synthetic data, default settings
    python part2.py --data path/to.npy     # custom data
    python part2.py --steps 1000 --lr 0.005
    python part2.py --record frames/
    python part2.py --ablate no_stl        # ablation: disable STL term
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
import optax

from config import (
    N, K, RANDOM_SEED,
    N_STEPS, LEARNING_RATE, LOG_EVERY,
    W_STL, W_SMOOTH, W_DEVIATION, W_SEPARATION,
    CONVERGE_RADIUS, DIVERGE_RADIUS, HOTSPOT_TIME_WINDOW,
    Hotspot,
)

# stljax imports — adjust to match the installed API version.
try:
    import stljax
    _STLJAX_OK = True
except ImportError:
    _STLJAX_OK = False
    print("[warning] stljax not found; STL robustness will return 0.")


# ═══════════════════════════════════════════════════════════════════════════
# STL specifications
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# A) Start / end correctness
# ---------------------------------------------------------------------------

def robustness_start_end(
    pos: Any,
    traj_in: Any,
    tolerance: float = 20.0,
) -> Any:
    """Robustness: each object starts and ends within ``tolerance`` of its input.

    Args:
        pos:       JAX array, shape (K, N, 2) — decision variable.
        traj_in:   JAX array, shape (N, K, 2) — reference trajectories.
        tolerance: Maximum allowed displacement in pixels.

    Returns:
        Scalar robustness value (higher = better satisfied).
    """
    # TODO (Part 2 — A): Implement using stljax or a plain differentiable expression.
    #
    #   stljax approach:
    #     Define a predicate: dist(pos[0, i], traj_in[i, 0]) < tolerance
    #     Use stljax.Always or stljax.Eventually over a time window.
    #
    #   Differentiable fallback (no stljax required):
    #     start_err = jnp.sqrt(((pos[0] - traj_in[:, 0, :])**2).sum(-1))   # (N,)
    #     end_err   = jnp.sqrt(((pos[-1] - traj_in[:, -1, :])**2).sum(-1)) # (N,)
    #     rho = tolerance - jnp.maximum(start_err, end_err)
    #     return rho.min()   # worst-case object robustness
    #
    # Placeholder: return 0.
    return jnp.array(0.0)


# ---------------------------------------------------------------------------
# B) Hotspot semantics
# ---------------------------------------------------------------------------

def robustness_convergence(
    pos: Any,
    hotspot: Hotspot,
    radius: float = CONVERGE_RADIUS,
    time_window: tuple[int, int] = HOTSPOT_TIME_WINDOW,
) -> Any:
    """Robustness: objects in hotspot.group eventually reach the hotspot.

    "Eventually" is evaluated over ``time_window`` = [t_lo, t_hi].

    Args:
        pos:         JAX array, shape (K, N, 2).
        hotspot:     Convergence hotspot.
        radius:      Tolerance radius around the hotspot centre.
        time_window: (t_lo, t_hi) frame range for the Eventually operator.

    Returns:
        Scalar robustness.
    """
    # TODO (Part 2 — B): Implement convergence STL spec.
    #
    #   Intent: "Eventually in [t_lo, t_hi], all objects in hotspot.group
    #            are within radius of (hotspot.x, hotspot.y)."
    #
    #   stljax approach:
    #     pred = lambda t: radius - dist(pos[t, group], hotspot_pt)
    #     rho  = stljax.Eventually(pred, interval=time_window)
    #
    #   Differentiable fallback:
    #     t_lo, t_hi = time_window
    #     dists = [jnp.sqrt(((pos[t, group] - hp)**2).sum(-1)).max()
    #              for t in range(t_lo, t_hi + 1)]   # worst object each frame
    #     rho = radius - jnp.stack(dists).min()       # best frame (Eventually)
    #     return rho
    #
    # Placeholder:
    return jnp.array(0.0)


def robustness_divergence(
    pos: Any,
    hotspot: Hotspot,
    min_spread: float = DIVERGE_RADIUS,
    time_window: tuple[int, int] | None = None,
) -> Any:
    """Robustness: after the divergence hotspot, objects spread apart.

    Args:
        pos:         JAX array, shape (K, N, 2).
        hotspot:     Divergence hotspot.
        min_spread:  Minimum required spread (std) among group members.
        time_window: (t_lo, t_hi) frames; defaults to [hotspot.time_step, K-1].

    Returns:
        Scalar robustness.
    """
    # TODO (Part 2 — B): Implement divergence STL spec.
    #
    #   Intent: "After t_diverge, the group of objects eventually
    #            becomes spread out (pairwise distance > min_spread)."
    #
    #   Differentiable fallback:
    #     t_lo = hotspot.time_step
    #     t_hi = K - 1
    #     group_pos = pos[t_lo:t_hi+1][:, hotspot.group, :]  # (T, n_g, 2)
    #     # Compute pairwise distances at each frame, take min over pairs.
    #     # Robustness = min_spread - worst_case_min_pairwise_dist
    #
    # Placeholder:
    return jnp.array(0.0)


# ---------------------------------------------------------------------------
# Combined STL robustness
# ---------------------------------------------------------------------------

def total_stl_robustness(
    pos: Any,
    traj_in: Any,
    hotspots: list[Hotspot],
    tolerance: float = 20.0,
) -> Any:
    """Sum all STL robustness terms.

    Args:
        pos:      JAX array, shape (K, N, 2).
        traj_in:  JAX array, shape (N, K, 2).
        hotspots: List of Hotspot objects.
        tolerance: Start/end tolerance.

    Returns:
        Scalar total STL robustness (maximise this).
    """
    rho = robustness_start_end(pos, traj_in, tolerance)

    for h in hotspots:
        if h.kind == "converge":
            rho = rho + robustness_convergence(pos, h)
        elif h.kind == "diverge":
            rho = rho + robustness_divergence(pos, h)

    return rho


# ═══════════════════════════════════════════════════════════════════════════
# Differentiable losses / regularisers
# ═══════════════════════════════════════════════════════════════════════════

def smoothness_loss(pos: Any) -> Any:
    """Velocity + acceleration penalty.

    Args:
        pos: JAX array, shape (K, N, 2).

    Returns:
        Scalar penalty (lower = smoother).
    """
    vel = jnp.diff(pos, axis=0)        # (K-1, N, 2)
    acc = jnp.diff(vel, axis=0)        # (K-2, N, 2)
    return (vel ** 2).mean() + (acc ** 2).mean()


def deviation_loss(pos: Any, traj_in: Any) -> Any:
    """Mean squared deviation from the input trajectories.

    Args:
        pos:     JAX array, shape (K, N, 2).
        traj_in: JAX array, shape (N, K, 2).

    Returns:
        Scalar penalty.
    """
    # traj_in is (N, K, 2); transpose to (K, N, 2) for comparison.
    ref = jnp.transpose(traj_in, (1, 0, 2))  # (K, N, 2)
    return ((pos - ref) ** 2).mean()


def separation_loss(pos: Any, min_dist: float = 8.0) -> Any:
    """Encourage pairwise separation to reduce occlusion.

    Penalises pairs of objects that are closer than ``min_dist``.

    Args:
        pos:      JAX array, shape (K, N, 2).
        min_dist: Minimum desired separation in pixels.

    Returns:
        Scalar penalty.
    """
    # TODO (Part 2 — C): Implement the separation penalty.
    #
    #   Differentiable approach:
    #     diff = pos[:, :, None, :] - pos[:, None, :, :]   # (K, N, N, 2)
    #     dist2 = (diff**2).sum(-1)                         # (K, N, N)
    #     # Penalty: relu(min_dist^2 - dist2), upper triangle only
    #     penalty = jnp.maximum(0.0, min_dist**2 - dist2)
    #     # Zero out diagonal
    #     mask = 1 - jnp.eye(pos.shape[1])                  # (N, N)
    #     return (penalty * mask).mean()
    #
    # Placeholder:
    return jnp.array(0.0)


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------

def total_loss(
    pos: Any,
    traj_in: Any,
    hotspots: list[Hotspot],
    w_stl: float = W_STL,
    w_smooth: float = W_SMOOTH,
    w_deviation: float = W_DEVIATION,
    w_separation: float = W_SEPARATION,
) -> Any:
    """Compute the scalar optimisation objective.

    We *minimise* this value, so STL robustness is negated.

    Args:
        pos:         JAX array, shape (K, N, 2) — decision variables.
        traj_in:     JAX array, shape (N, K, 2) — input reference.
        hotspots:    Hotspot list for STL specs.
        w_*:         Loss term weights.

    Returns:
        Scalar loss.
    """
    stl_rho    = total_stl_robustness(pos, traj_in, hotspots)
    l_smooth   = smoothness_loss(pos)
    l_deviation = deviation_loss(pos, traj_in)
    l_sep      = separation_loss(pos)

    # Negate STL robustness (we minimise -> maximise robustness).
    loss = (
        -w_stl * stl_rho
        + w_smooth * l_smooth
        + w_deviation * l_deviation
        + w_separation * l_sep
    )
    return loss


# ═══════════════════════════════════════════════════════════════════════════
# Optimisation loop
# ═══════════════════════════════════════════════════════════════════════════

def optimize(
    traj_in: np.ndarray,
    init_pos: np.ndarray | None = None,
    hotspots: list[Hotspot] | None = None,
    n_steps: int = N_STEPS,
    lr: float = LEARNING_RATE,
    log_every: int = LOG_EVERY,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Optimise ``pos`` to minimise the combined loss.

    Args:
        traj_in:   Shape (N, K, 2) NumPy array — input reference.
        init_pos:  Shape (K, N, 2) NumPy array — initial guess.
                   Defaults to the transposed input (no bundling).
        hotspots:  Hotspot objects for the STL specs.
        n_steps:   Number of gradient steps.
        lr:        Adam learning rate.
        log_every: Print/record diagnostics every this many steps.

    Returns:
        pos_opt:  Shape (K, N, 2) optimised positions (NumPy, float32).
        history:  Dict with keys ``"loss"`` and ``"step"`` (lists).
    """
    hotspots = hotspots or []

    if init_pos is None:
        init_pos = traj_in.transpose(1, 0, 2).copy()  # (K, N, 2)

    # Convert to JAX arrays.
    traj_jax = jnp.array(traj_in, dtype=jnp.float32)     # (N, K, 2)
    pos      = jnp.array(init_pos, dtype=jnp.float32)    # (K, N, 2)

    # --- Optimiser ---
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(pos)

    # --- JIT-compiled update step ---
    @jax.jit
    def step(pos, opt_state):
        loss_val, grads = jax.value_and_grad(total_loss)(
            pos, traj_jax, hotspots
        )
        updates, opt_state_new = optimizer.update(grads, opt_state)
        pos_new = optax.apply_updates(pos, updates)
        return pos_new, opt_state_new, loss_val

    history: dict[str, list[float]] = {"loss": [], "step": []}

    print(f"Starting optimisation: {n_steps} steps, lr={lr}")
    for i in range(n_steps):
        pos, opt_state, loss_val = step(pos, opt_state)

        if i % log_every == 0 or i == n_steps - 1:
            lv = float(loss_val)
            history["loss"].append(lv)
            history["step"].append(i)
            print(f"  step {i:5d} / {n_steps}   loss = {lv:.4f}")

    pos_opt = np.array(pos, dtype=np.float32)
    return pos_opt, history


def plot_convergence(history: dict[str, list[float]], save_path: str | None = None) -> None:
    """Plot the loss curve over optimisation steps.

    Args:
        history:   Dict returned by :func:`optimize`.
        save_path: If given, save the figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["loss"], lw=2)
    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Loss")
    ax.set_title("Part 2 — Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved convergence plot -> {save_path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Run Part 2 pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _print_metrics(traj_in: np.ndarray, pos: np.ndarray) -> None:
    import metrics
    print(f"  occlusion      : {metrics.occlusion_rate(pos):.4f}")
    print(f"  mean deviation : {metrics.mean_deviation(traj_in, pos):.2f} px")
    sm = metrics.smoothness_score(pos)
    print(f"  mean velocity  : {sm['mean_velocity']:.2f}")
    print(f"  mean accel     : {sm['mean_acceleration']:.2f}")
    print(f"  dispersion     : {metrics.mean_dispersion(pos):.2f}")


def run(
    data_path: str | None = None,
    n_steps: int = N_STEPS,
    lr: float = LEARNING_RATE,
    record_dir: str | None = None,
    save_pos: str | None = None,
    ablate: str | None = None,
) -> None:
    from utils import set_global_seed
    from data import generate_dataset, load_trajectories, save_positions
    from part1 import baseline, bundle_trajectories, assign_offsets

    set_global_seed(RANDOM_SEED)

    # --- Data ---
    if data_path is None:
        print("Generating synthetic dataset ...")
        traj_in, hotspots = generate_dataset(n=N, k=K)
    else:
        print(f"Loading {data_path} ...")
        traj_in = load_trajectories(data_path, resample_to=K)
        hotspots = []

    print(f"traj_in: {traj_in.shape}")

    # --- Warm-start from Part 1 ---
    bundled, groups = bundle_trajectories(traj_in, hotspots=hotspots)
    init_pos = assign_offsets(bundled, groups)   # (K, N, 2)

    # --- Ablation weight overrides ---
    w_stl = 0.0 if ablate == "no_stl" else W_STL
    w_smooth = 0.0 if ablate == "no_smooth" else W_SMOOTH
    w_dev = 0.0 if ablate == "no_deviation" else W_DEVIATION
    w_sep = 0.0 if ablate == "no_separation" else W_SEPARATION

    if ablate:
        print(f"[ablation] disabled term: {ablate}")

    # --- Optimise ---
    pos_opt, history = optimize(
        traj_in,
        init_pos=init_pos,
        hotspots=hotspots,
        n_steps=n_steps,
        lr=lr,
    )

    # --- Evaluate ---
    pos_baseline = baseline(traj_in)

    print("\n--- Baseline (no processing) ---")
    _print_metrics(traj_in, pos_baseline)

    print("\n--- Part 1 (bundled + layout) ---")
    _print_metrics(traj_in, init_pos)

    print("\n--- Part 2 (optimised) ---")
    _print_metrics(traj_in, pos_opt)

    # --- Plots / output ---
    plot_convergence(history, save_path="reports/figures/convergence.png")

    if save_pos:
        save_positions(pos_opt, save_pos)

    # --- Render ---
    if record_dir is not None:
        from viewer import export_frames
        export_frames(pos_opt, record_dir)
    else:
        from viewer import play
        play(pos_opt, title="Part 2 — Optimised Animation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Part 2 optimisation pipeline")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="Save optimised pos to .npy")
    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        choices=["no_stl", "no_smooth", "no_deviation", "no_separation"],
        help="Disable one loss term for ablation study",
    )
    args = parser.parse_args()

    run(
        data_path=args.data,
        n_steps=args.steps,
        lr=args.lr,
        record_dir=args.record,
        save_pos=args.save,
        ablate=args.ablate,
    )
