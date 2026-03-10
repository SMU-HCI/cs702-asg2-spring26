"""Part 2 — Bayesian Filtering for Improved Hand Tracking.

Loads data recorded by ``part1.py --save`` and compares all four Bayesian
filters against the naïve predictor.  Optionally runs a live webcam session
with a selected filter overlaid on the MediaPipe output.

Usage::

    python part2.py --load data.npz            # offline RMSE comparison table
    python part2.py --live --filter kf         # live capture with KF overlay
    python part2.py --live --filter pf         # live capture with PF overlay
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from filters import (
    Filter,
    KalmanFilter,
    ParticleFilter,
    MovingHorizonEstimator,
)
from part1 import (
    INDEX_TIP, TRACKED,
    C_RAW, C_PRED, C_TEXT,
    PREDICTION_HORIZONS_MS,
    evaluate_rmse,
    world_to_pixel,
)

FILTER_REGISTRY: dict[str, type[Filter]] = {
    "kf":  KalmanFilter,
    "pf":  ParticleFilter,
    "mhe": MovingHorizonEstimator,
}


# ═══════════════════════════════════════════════════════════════════════════
# Offline evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_filter(
    filter_: Filter,
    positions: np.ndarray,
    timestamps: np.ndarray,
    horizons_ms: list[int] = PREDICTION_HORIZONS_MS,
) -> dict[int, float]:
    """Run a filter on pre-recorded data and return RMSE per horizon.

    Args:
        filter_:     An uninitialised Filter instance.
        positions:   Ground-truth world positions, shape (T, 3).
        timestamps:  Capture timestamps in seconds, shape (T,).
        horizons_ms: Prediction horizons to evaluate.

    Returns:
        Dict mapping horizon (ms) → RMSE in metres.
    """
    errors: dict[int, list[float]] = {h: [] for h in horizons_ms}
    prev_t: float | None = None

    for t, (pos, ts) in enumerate(zip(positions, timestamps)):
        dt = (ts - prev_t) if prev_t is not None else 1 / 30
        prev_t = ts

        filter_.predict(dt)
        filter_.update(pos)

        for h in horizons_ms:
            future_idx = np.searchsorted(timestamps, ts + h / 1000.0)
            if future_idx >= len(positions):
                continue
            predicted = filter_.predict_ahead(h)
            errors[h].append(float(np.linalg.norm(predicted - positions[future_idx])))

    return {h: float(np.mean(v)) for h, v in errors.items() if v}


def run_evaluate(load_path: Path) -> None:
    """Load saved data, run all filters, and print a comparison table."""
    data = np.load(load_path)
    positions  = data["positions"]   # (T, 3)
    timestamps = data["timestamps"]  # (T,)
    print(f"Loaded {len(positions)} frames from {load_path}\n")

    # Naïve baseline
    results: dict[str, dict[int, float]] = {
        "naive": evaluate_rmse(positions, timestamps),
    }

    for name, cls in FILTER_REGISTRY.items():
        try:
            results[name] = evaluate_filter(cls(), positions, timestamps)
        except NotImplementedError:
            results[name] = {h: float("nan") for h in PREDICTION_HORIZONS_MS}

    # Print table
    col = "".join(f"  {h}ms (cm)" for h in PREDICTION_HORIZONS_MS)
    header = f"{'Method':<8}{col}"
    print(header)
    print("─" * len(header))
    for name, rmse in results.items():
        row = f"{name:<8}"
        for h in PREDICTION_HORIZONS_MS:
            v = rmse.get(h, float("nan"))
            row += f"  {'  nan':>9}" if np.isnan(v) else f"  {v * 100:>8.2f}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════════
# Live capture
# ═══════════════════════════════════════════════════════════════════════════

def run_live(filter_name: str) -> None:
    """Run live MediaPipe tracking with a selected Bayesian filter overlaid.

    Args:
        filter_name: Key from FILTER_REGISTRY (``kf``, ``pf``, ``mhe``).
    """
    if filter_name not in FILTER_REGISTRY:
        raise ValueError(
            f"Unknown filter '{filter_name}'. Choose from: {list(FILTER_REGISTRY)}"
        )

    active_filter: Filter = FILTER_REGISTRY[filter_name]()
    print(f"Running live with {filter_name.upper()} filter. Press q / Esc to quit.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (device 0).")

    prev_time: float | None = None
    start_time = time.perf_counter()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.perf_counter() - start_time
            dt = (now - prev_time) if prev_time is not None else 1 / 30
            prev_time = now

            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_lms, hand_world_lms in zip(
                    results.multi_hand_landmarks,
                    results.multi_hand_world_landmarks,
                ):
                    mp_drawing.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    wlm = hand_world_lms.landmark[INDEX_TIP]
                    pos_world = np.array([wlm.x, wlm.y, wlm.z])

                    active_filter.predict(dt)
                    active_filter.update(pos_world)

                    # Draw filter prediction circles
                    for horizon_ms, color in C_PRED.items():
                        pred = active_filter.predict_ahead(horizon_ms)
                        px, py = world_to_pixel(
                            pred,
                            hand_lms.landmark[INDEX_TIP],
                            hand_world_lms.landmark[INDEX_TIP],
                            img_w, img_h,
                        )
                        cv2.circle(frame, (px, py), 6, color, -1)
                        cv2.putText(frame, f"{horizon_ms}ms", (px + 8, py + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    for lm_id in TRACKED:
                        lm = hand_lms.landmark[lm_id]
                        cv2.circle(frame,
                                   (int(lm.x * img_w), int(lm.y * img_h)),
                                   8, C_RAW, -1)

            cv2.putText(frame, f"Filter: {filter_name.upper()}  |  q: quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)
            cv2.imshow("Part 2 – Bayesian Filtering", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", metavar="FILE",
                        help="NPZ file recorded by part1.py; print RMSE table")
    parser.add_argument("--live", action="store_true",
                        help="Run live webcam capture with a filter overlay")
    parser.add_argument("--filter", default="kf", choices=list(FILTER_REGISTRY),
                        help="Filter to use in live mode (default: kf)")
    args = parser.parse_args()

    if args.live:
        run_live(args.filter)
    elif args.load:
        run_evaluate(Path(args.load))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
