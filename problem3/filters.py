"""Bayesian filter implementations for Part 2.

All filters share a common interface::

    filter.predict(dt)            – time-update step (prior)
    filter.update(y)              – measurement-update step (posterior)
    filter.get_state()            – current position estimate, shape (3,)
    filter.predict_ahead(tau_ms)  – predicted future position, shape (3,)

State convention for the CV model (6-dimensional)::

    x = [px, vx, py, vy, pz, vz]^T
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Base interface
# ═══════════════════════════════════════════════════════════════════════════

class Filter(ABC):
    """Common interface for all state estimators."""

    @abstractmethod
    def predict(self, dt: float) -> None:
        """Time-update step: propagate state forward by dt seconds."""

    @abstractmethod
    def update(self, y: np.ndarray) -> None:
        """Measurement-update step: incorporate observation y, shape (3,)."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current position estimate, shape (3,)."""

    @abstractmethod
    def predict_ahead(self, tau_ms: float) -> np.ndarray:
        """Predict position tau_ms milliseconds ahead, shape (3,)."""


# ═══════════════════════════════════════════════════════════════════════════
# Kalman Filter (KF)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KalmanFilter(Filter):
    """Discrete-time Kalman filter with the constant-velocity (CV) model.

    State vector:   x = [px, vx, py, vy, pz, vz]^T   shape (6,)
    Observation:    y = [px, py, pz]^T                shape (3,)

    System matrices
    ---------------
    F  (6×6)  state transition  –  see :meth:`_make_F`
    H  (3×6)  observation       –  see :meth:`_make_H`
    Q  (6×6)  process noise     –  see :meth:`_make_Q`
    R  (3×3)  measurement noise –  see :meth:`_make_R`

    Tuning
    ------
    q  – process noise scale.  Increase if the filter lags fast motion.
    r  – measurement noise scale.  Estimate from a static recording:
         hold the hand still for ~10 s and compute per-axis variance.
    """

    q: float = 10.0   # process noise scale
    r: float = 1e-4   # measurement noise scale

    _x: np.ndarray = field(default_factory=lambda: np.zeros(6), repr=False)
    _P: np.ndarray = field(default_factory=lambda: np.eye(6),   repr=False)
    _initialized: bool = field(default=False, repr=False)

    # ── System matrices ────────────────────────────────────────────────

    def _make_F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 1] = dt  # px += vx · dt
        F[2, 3] = dt  # py += vy · dt
        F[4, 5] = dt  # pz += vz · dt
        return F

    def _make_H(self) -> np.ndarray:
        H = np.zeros((3, 6))
        H[0, 0] = 1.0  # measure px
        H[1, 2] = 1.0  # measure py
        H[2, 4] = 1.0  # measure pz
        return H

    def _make_Q(self, dt: float) -> np.ndarray:
        # TODO: A principled choice uses the continuous white-noise
        # acceleration model integrated over dt.  Replace this simple
        # diagonal approximation if desired.
        Q = np.zeros((6, 6))
        Q[0, 0] = Q[2, 2] = Q[4, 4] = self.q * dt ** 2
        Q[1, 1] = Q[3, 3] = Q[5, 5] = self.q
        return Q

    def _make_R(self) -> np.ndarray:
        return np.eye(3) * self.r

    # ── Filter steps ───────────────────────────────────────────────────

    def predict(self, dt: float) -> None:
        """Kalman predict step.

        TODO: Propagate state and covariance forward::

            x̂⁻ = F · x̂
            P⁻  = F · P · Fᵀ + Q
        """
        if not self._initialized:
            return
        F = self._make_F(dt)
        Q = self._make_Q(dt)

        # TODO: implement – replace the lines below.
        # self._x = ...
        # self._P = ...
        raise NotImplementedError("KalmanFilter.predict() is not yet implemented.")

    def update(self, y: np.ndarray) -> None:
        """Kalman update step.

        Args:
            y: Observed position, shape (3,).

        TODO: Incorporate the measurement::

            z = y − H · x̂⁻                  (innovation)
            S = H · P⁻ · Hᵀ + R              (innovation covariance)
            K = P⁻ · Hᵀ · S⁻¹               (Kalman gain)
            x̂ = x̂⁻ + K · z
            P = (I − K · H) · P⁻
        """
        if not self._initialized:
            self._x = np.zeros(6)
            self._x[[0, 2, 4]] = y   # position initialised from first measurement
            self._P = np.eye(6)
            self._initialized = True
            return

        H = self._make_H()
        R = self._make_R()

        # TODO: implement – replace the line below.
        raise NotImplementedError("KalmanFilter.update() is not yet implemented.")

    # ── Output ─────────────────────────────────────────────────────────

    def get_state(self) -> np.ndarray:
        return self._x[[0, 2, 4]].copy()

    def predict_ahead(self, tau_ms: float) -> np.ndarray:
        """Predict position tau_ms ms ahead.

        TODO: Propagate through F for n = ceil(tau / dt) steps rather than
              the linear shortcut below.
        """
        if not self._initialized:
            return np.zeros(3)
        tau_s = tau_ms / 1000.0
        # Linear shortcut – replace with matrix propagation.
        return np.array([
            self._x[0] + self._x[1] * tau_s,
            self._x[2] + self._x[3] * tau_s,
            self._x[4] + self._x[5] * tau_s,
        ])


# ═══════════════════════════════════════════════════════════════════════════
# Extended Kalman Filter (EKF)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExtendedKalmanFilter(Filter):
    """Extended Kalman Filter for nonlinear dynamics or observations.

    The EKF replaces the linear F and H matrices with their Jacobians J_f
    and J_h evaluated at the current state estimate::

        Predict:  x̂⁻ = f(x̂),         P⁻ = J_f · P · J_fᵀ + Q
        Update:   z   = y − h(x̂⁻),   S  = J_h · P⁻ · J_hᵀ + R
                  K   = P⁻ · J_hᵀ · S⁻¹
                  x̂   = x̂⁻ + K · z,  P  = (I − K · J_h) · P⁻

    TODO: Choose a nonlinear extension.  Options include:
      - Polar-coordinate observations (angle + distance to a target point).
      - Joint-angle kinematics from multiple MediaPipe landmarks.
    Then implement f(), h(), J_f(), J_h(), predict(), and update().
    """

    q: float = 10.0
    r: float = 1e-4

    _x: np.ndarray = field(default_factory=lambda: np.zeros(6), repr=False)
    _P: np.ndarray = field(default_factory=lambda: np.eye(6),   repr=False)
    _initialized: bool = field(default=False, repr=False)

    def predict(self, dt: float) -> None:
        raise NotImplementedError("ExtendedKalmanFilter.predict() — TODO")

    def update(self, y: np.ndarray) -> None:
        if not self._initialized:
            self._x = np.zeros(6)
            self._x[[0, 2, 4]] = y
            self._P = np.eye(6)
            self._initialized = True
            return
        raise NotImplementedError("ExtendedKalmanFilter.update() — TODO")

    def get_state(self) -> np.ndarray:
        return self._x[[0, 2, 4]].copy()

    def predict_ahead(self, tau_ms: float) -> np.ndarray:
        if not self._initialized:
            return np.zeros(3)
        raise NotImplementedError("ExtendedKalmanFilter.predict_ahead() — TODO")


# ═══════════════════════════════════════════════════════════════════════════
# Particle Filter (PF)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParticleFilter(Filter):
    """Sequential Monte Carlo (particle filter) for non-Gaussian noise.

    Represents the posterior p(x_k | y_{1:k}) as a weighted set of N_p
    particles {x^(i), w^(i)}.

    Algorithm
    ---------
    Predict:  x^(i) ← f(x^(i), ε^(i))      sample from process model
    Update:   w^(i) ← p(y | x^(i))          weight by likelihood
              Normalise weights (log-sum-exp for numerical stability).
    Resample: when N_eff = 1 / Σ(w^(i))² < n_particles / 2, resample
              (systematic resampling recommended).

    TODO: Implement predict(), update(), and the resampling step.
    """

    n_particles: int = 200
    q: float = 0.05   # process noise std dev (metres / step)
    r: float = 0.01   # measurement noise std dev (metres)

    _particles: np.ndarray = field(init=False, repr=False)  # (N_p, 6)
    _weights:   np.ndarray = field(init=False, repr=False)  # (N_p,)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._particles = np.zeros((self.n_particles, 6))
        self._weights   = np.ones(self.n_particles) / self.n_particles

    def _init_particles(self, y: np.ndarray) -> None:
        """Spread particles around the first measurement."""
        self._particles[:, [0, 2, 4]] = y + np.random.randn(self.n_particles, 3) * self.r
        self._particles[:, [1, 3, 5]] = 0.0
        self._weights[:] = 1.0 / self.n_particles
        self._initialized = True

    def predict(self, dt: float) -> None:
        """Propagate particles through the CV transition and add process noise.

        TODO: For each particle x^(i):
                  x^(i) = F · x^(i) + noise,   noise ~ N(0, Q)
        """
        if not self._initialized:
            return
        raise NotImplementedError("ParticleFilter.predict() — TODO")

    def update(self, y: np.ndarray) -> None:
        """Weight particles by Gaussian likelihood p(y | x^(i)).

        TODO:
        1. If not initialised, call ``_init_particles(y)`` and return.
        2. Compute log-likelihood for each particle::

               log w^(i) += −0.5 · ‖y − H x^(i)‖² / r²

        3. Normalise (use log-sum-exp).
        4. Resample when N_eff < n_particles / 2.
        """
        if not self._initialized:
            self._init_particles(y)
            return
        raise NotImplementedError("ParticleFilter.update() — TODO")

    def get_state(self) -> np.ndarray:
        """Return the weighted mean position, shape (3,)."""
        if not self._initialized:
            return np.zeros(3)
        mean = (self._weights[:, None] * self._particles).sum(axis=0)
        return mean[[0, 2, 4]]

    def predict_ahead(self, tau_ms: float) -> np.ndarray:
        """Return the weighted mean after propagating particles forward.

        TODO: Propagate each particle through F for n steps, then return
              the weighted mean position.
        """
        if not self._initialized:
            return np.zeros(3)
        raise NotImplementedError("ParticleFilter.predict_ahead() — TODO")


# ═══════════════════════════════════════════════════════════════════════════
# Moving Horizon Estimator (MHE)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MovingHorizonEstimator(Filter):
    """Moving Horizon Estimation: MAP estimation over a sliding window.

    At each time step, solve the batch optimisation problem::

        min_{x_{k-N:k}}  Σ_{t=k-N}^{k}   ‖y_t − H x_t‖²_{R⁻¹}
                       + Σ_{t=k-N}^{k-1} ‖x_{t+1} − F x_t‖²_{Q⁻¹}
                       + ‖x_{k-N} − x̄_{k-N}‖²_{P₀⁻¹}

    where N is the horizon length.

    TODO: Implement the optimisation solve.  You may use
    ``scipy.optimize.minimize`` or formulate as a banded least-squares
    problem (``scipy.linalg.solve_banded`` or ``numpy.linalg.lstsq``).
    """

    horizon: int = 10   # number of past time steps in the window
    q: float = 10.0
    r: float = 1e-4

    _obs_buffer: list = field(default_factory=list, repr=False)
    _dt_buffer:  list = field(default_factory=list, repr=False)
    _x_est: np.ndarray = field(default_factory=lambda: np.zeros(6), repr=False)
    _initialized: bool = field(default=False, repr=False)

    def predict(self, dt: float) -> None:
        """Buffer the time step; the solve happens inside update()."""
        self._dt_buffer.append(dt)
        if len(self._dt_buffer) > self.horizon:
            self._dt_buffer.pop(0)

    def update(self, y: np.ndarray) -> None:
        """Add measurement to the window and solve.

        TODO:
        1. Append y to ``_obs_buffer``; trim to ``horizon`` entries.
        2. Build the stacked least-squares system over the window.
        3. Solve and store the most-recent state in ``_x_est``.
        """
        self._obs_buffer.append(y.copy())
        if len(self._obs_buffer) > self.horizon:
            self._obs_buffer.pop(0)

        if not self._initialized:
            self._x_est = np.zeros(6)
            self._x_est[[0, 2, 4]] = y
            self._initialized = True
            return

        raise NotImplementedError("MovingHorizonEstimator.update() — TODO")

    def get_state(self) -> np.ndarray:
        return self._x_est[[0, 2, 4]].copy()

    def predict_ahead(self, tau_ms: float) -> np.ndarray:
        if not self._initialized:
            return np.zeros(3)
        tau_s = tau_ms / 1000.0
        # Linear shortcut – replace with matrix propagation once MHE is implemented.
        return np.array([
            self._x_est[0] + self._x_est[1] * tau_s,
            self._x_est[2] + self._x_est[3] * tau_s,
            self._x_est[4] + self._x_est[5] * tau_s,
        ])
