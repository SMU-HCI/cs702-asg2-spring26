"""Part 1 — Kinematics-based Hand Tracking with MediaPipe.

Captures hand landmark positions from a webcam, visualises them in real time,
and uses a naïve kinematic model to predict future hand positions.

Usage::

    python part1.py                     # live capture + visualise predictions
    python part1.py --save data.npz     # record a session to file
    python part1.py --load data.npz     # replay saved data and evaluate RMSE
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# MediaPipe hand landmark indices (see full list at
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
WRIST     = 0
THUMB_TIP = 4
INDEX_TIP = 8
TRACKED   = [WRIST, INDEX_TIP, THUMB_TIP]

PREDICTION_HORIZONS_MS = [100, 200, 300]

# BGR display colours
C_RAW      = (0, 220, 0)    # green  – raw MediaPipe position
C_PRED     = {              # prediction dots per horizon (yellow → red)
    100: (0, 200, 255),
    200: (0, 120, 255),
    300: (0,  50, 200),
}
C_TEXT = (255, 255, 255)

# Pixel-per-metre scale used when projecting world-coord predictions to screen
WORLD_TO_PX_SCALE = 400


# ═══════════════════════════════════════════════════════════════════════════
# Kinematic state-space model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KinematicState:
    """State for the constant-velocity (CV) model (one landmark, 3-D).

    State per axis: [p, v]^T.
    For the CA model, extend to [p, v, a]^T and add an acceleration field.
    """
    pos: np.ndarray  # shape (3,) – world x, y, z in metres
    vel: np.ndarray  # shape (3,) – velocity in m/s


class NaivePredictor:
    """Naïve kinematic predictor: forward-propagates the last estimated state.

    This class is intentionally incomplete.  Your tasks are:

    1. In ``update()``, replace the finite-difference placeholder with a
       proper state-space update (transition matrix A).
    2. In ``predict_ahead()``, replace the linear extrapolation with
       propagation through A for the correct number of steps.
    """

    def __init__(self, dt_default: float = 1 / 30) -> None:
        self.dt_default = dt_default
        self.state: Optional[KinematicState] = None
        self._prev_time: Optional[float] = None

    # ------------------------------------------------------------------

    def update(self, pos: np.ndarray, timestamp: float) -> None:
        """Update the state estimate from a new position measurement.

        Args:
            pos:       World-coordinate position of the landmark, shape (3,).
            timestamp: Capture time in seconds.

        TODO: Implement a proper state-space update using the transition
              matrix A.  For the CV model (per axis):

                  x_k = A x_{k-1} + ε_k,   A = [[1, Δt],
                                                  [0,  1 ]]

              The finite-difference velocity estimate below is a placeholder.
        """
        if self.state is None:
            self.state = KinematicState(pos=pos.copy(), vel=np.zeros(3))
            self._prev_time = timestamp
            return

        dt = timestamp - self._prev_time if self._prev_time is not None else self.dt_default
        dt = max(dt, 1e-6)

        # TODO: Replace with your state-space update.
        self.state.vel = (pos - self.state.pos) / dt
        self.state.pos = pos.copy()
        self._prev_time = timestamp

    # ------------------------------------------------------------------

    def predict_ahead(self, tau_ms: float) -> np.ndarray:
        """Predict the landmark position tau_ms milliseconds into the future.

        Args:
            tau_ms: Prediction horizon in milliseconds.

        Returns:
            Predicted world-coordinate position, shape (3,).

        TODO: Replace the linear extrapolation below with propagation through
              the transition matrix A for n = ceil(tau / Δt) steps:

                  x_future = A^n x_current
        """
        if self.state is None:
            return np.zeros(3)
        tau_s = tau_ms / 1000.0
        # TODO: Replace with matrix propagation.
        return self.state.pos + self.state.vel * tau_s


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_rmse(
    positions: np.ndarray,
    timestamps: np.ndarray,
    horizons_ms: list[int] = PREDICTION_HORIZONS_MS,
) -> dict[int, float]:
    """Evaluate naïve-predictor RMSE on pre-recorded data.

    Args:
        positions:   Ground-truth world positions, shape (T, 3).
        timestamps:  Capture timestamps in seconds, shape (T,).
        horizons_ms: Prediction horizons to evaluate.

    Returns:
        Dict mapping horizon (ms) → RMSE in metres.
    """
    predictor = NaivePredictor()
    errors: dict[int, list[float]] = {h: [] for h in horizons_ms}

    for t, (pos, ts) in enumerate(zip(positions, timestamps)):
        predictor.update(pos, ts)
        for h in horizons_ms:
            future_idx = np.searchsorted(timestamps, ts + h / 1000.0)
            if future_idx >= len(positions):
                continue
            predicted = predictor.predict_ahead(h)
            errors[h].append(float(np.linalg.norm(predicted - positions[future_idx])))

    return {h: float(np.mean(v)) for h, v in errors.items() if v}


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def world_to_pixel(
    world_pos: np.ndarray,
    ref_img_lm,
    ref_world_lm,
    img_w: int,
    img_h: int,
) -> tuple[int, int]:
    """Map a world-coordinate position to an approximate pixel location.

    Uses the corresponding image-space landmark as an anchor and projects
    the world-space displacement onto the image plane.

    Args:
        world_pos:    Predicted world position (x, y, z), shape (3,).
        ref_img_lm:   MediaPipe image-normalised landmark (the same joint).
        ref_world_lm: MediaPipe world-coordinate landmark (the same joint).
        img_w, img_h: Frame dimensions in pixels.

    Returns:
        (px, py) pixel coordinates, suitable for cv2 drawing.
    """
    ref_px = np.array([ref_img_lm.x * img_w, ref_img_lm.y * img_h])
    delta_world = world_pos[:2] - np.array([ref_world_lm.x, ref_world_lm.y])
    px = ref_px + delta_world * WORLD_TO_PX_SCALE
    return int(px[0]), int(px[1])


def draw_predictions(
    frame: np.ndarray,
    predictor: NaivePredictor,
    ref_img_lm,
    ref_world_lm,
) -> None:
    """Overlay prediction circles for each horizon on the frame (in-place)."""
    h, w = frame.shape[:2]
    for horizon_ms, color in C_PRED.items():
        pred = predictor.predict_ahead(horizon_ms)
        px, py = world_to_pixel(pred, ref_img_lm, ref_world_lm, w, h)
        cv2.circle(frame, (px, py), 6, color, -1)
        cv2.putText(frame, f"{horizon_ms}ms", (px + 8, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Live capture
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _get_hand_model() -> Path:
    """Download the hand landmarker model if not already present."""
    model_path = Path(__file__).parent / "hand_landmarker.task"
    if not model_path.exists():
        print("Downloading hand landmarker model…")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print(f"Saved model to {model_path}")
    return model_path


def run_live(save_path: Optional[Path] = None) -> None:
    """Run live MediaPipe hand tracking with naïve kinematic prediction.

    Args:
        save_path: If given, index-fingertip world positions are saved here
                   as a .npz file (keys: ``positions``, ``timestamps``).
    """
    model_path = _get_hand_model()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (device 0).")

    rec_positions: list[np.ndarray] = []
    rec_timestamps: list[float] = []

    predictor = NaivePredictor()
    start_time = time.perf_counter()

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = time.perf_counter() - start_time
            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            img_h, img_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(timestamp * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                for hand_lms, hand_world_lms in zip(
                    results.hand_landmarks,
                    results.hand_world_landmarks,
                ):
                    # Draw full hand skeleton
                    mp_vision.drawing_utils.draw_landmarks(
                        frame, hand_lms,
                        mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                        mp_vision.drawing_styles.get_default_hand_landmarks_style(),
                        mp_vision.drawing_styles.get_default_hand_connections_style(),
                    )

                    # Update predictor with index fingertip world position
                    wlm = hand_world_lms[INDEX_TIP]
                    pos_world = np.array([wlm.x, wlm.y, wlm.z])
                    predictor.update(pos_world, timestamp)

                    if save_path is not None:
                        rec_positions.append(pos_world.copy())
                        rec_timestamps.append(timestamp)

                    # Draw prediction circles for index fingertip
                    draw_predictions(
                        frame, predictor,
                        ref_img_lm=hand_lms[INDEX_TIP],
                        ref_world_lm=hand_world_lms[INDEX_TIP],
                    )

                    # Highlight tracked landmarks
                    for lm_id in TRACKED:
                        lm = hand_lms[lm_id]
                        px, py = int(lm.x * img_w), int(lm.y * img_h)
                        cv2.circle(frame, (px, py), 8, C_RAW, -1)

            # HUD
            status = f"Recording: {len(rec_timestamps)} frames" if save_path else "Press q to quit"
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)

            cv2.imshow("Part 1 – Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

    if save_path and rec_positions:
        np.savez(
            save_path,
            positions=np.array(rec_positions),
            timestamps=np.array(rec_timestamps),
        )
        print(f"Saved {len(rec_positions)} frames to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Offline evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluate(load_path: Path) -> None:
    """Load saved data, run the naïve predictor, and print RMSE results."""
    data = np.load(load_path)
    positions  = data["positions"]   # (T, 3)
    timestamps = data["timestamps"]  # (T,)
    print(f"Loaded {len(positions)} frames from {load_path}")

    rmse = evaluate_rmse(positions, timestamps)
    print("\nNaïve predictor RMSE:")
    for horizon, error in sorted(rmse.items()):
        print(f"  {horizon:>4} ms  →  {error * 100:.2f} cm")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="FILE",
                        help="Save recorded index-fingertip data to .npz file")
    parser.add_argument("--load", metavar="FILE",
                        help="Load saved data and evaluate RMSE")
    args = parser.parse_args()

    if args.load:
        run_evaluate(Path(args.load))
    else:
        run_live(save_path=Path(args.save) if args.save else None)


if __name__ == "__main__":
    main()
