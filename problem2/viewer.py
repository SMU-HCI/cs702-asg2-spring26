"""Pygame-based animation viewer and frame exporter.

Controls:
  SPACE   — pause / resume
  R       — restart from frame 0
  ESC / Q — quit
  LEFT    — step back one frame (while paused)
  RIGHT   — step forward one frame (while paused)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

import pygame

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS, DOT_RADIUS, BG_COLOR


# Colour palette — one colour per object (wraps around if N > len(PALETTE)).
PALETTE = [
    (231,  76,  60), ( 52, 152, 219), ( 46, 204, 113), (155,  89, 182),
    (243, 156,  18), ( 26, 188, 156), (230, 126,  34), ( 52,  73,  94),
    (149, 165, 166), (192,  57,  43), ( 41, 128, 185), ( 39, 174,  96),
    (142,  68, 173), (211,  84,   0), ( 22, 160, 133), (127, 140, 141),
]


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

def play(
    pos: NDArray,
    title: str = "Trajectory Animation",
    width: int = WINDOW_WIDTH,
    height: int = WINDOW_HEIGHT,
    fps: int = FPS,
    trail_length: int = 10,
) -> None:
    """Play the animation and block until the window is closed.

    Args:
        pos:          Shape (K, N, 2) — output positions (time-major).
        title:        Window title string.
        width:        Window width in pixels.
        height:       Window height in pixels.
        fps:          Playback frame rate.
        trail_length: Number of past frames to draw as a fading trail.
    """
    K, N, _ = pos.shape

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    frame = 0
    paused = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    frame = 0
                elif event.key == pygame.K_RIGHT and paused:
                    frame = min(frame + 1, K - 1)
                elif event.key == pygame.K_LEFT and paused:
                    frame = max(frame - 1, 0)

        # --- Draw ---
        screen.fill(BG_COLOR)

        # Draw trails.
        for obj in range(N):
            color = PALETTE[obj % len(PALETTE)]
            t_start = max(0, frame - trail_length)
            for t in range(t_start, frame):
                alpha = int(255 * (t - t_start) / max(trail_length, 1))
                trail_color = tuple(int(c * alpha / 255) for c in color)
                x, y = int(pos[t, obj, 0]), int(pos[t, obj, 1])
                pygame.draw.circle(screen, trail_color, (x, y), max(1, DOT_RADIUS - 2))

        # Draw current dots.
        for obj in range(N):
            color = PALETTE[obj % len(PALETTE)]
            x, y = int(pos[frame, obj, 0]), int(pos[frame, obj, 1])
            pygame.draw.circle(screen, color, (x, y), DOT_RADIUS)

        # HUD.
        status = "PAUSED" if paused else "PLAYING"
        hud = font.render(
            f"Frame {frame + 1}/{K}   {status}   (SPACE=pause  R=restart  ESC=quit)",
            True,
            (80, 80, 80),
        )
        screen.blit(hud, (10, 10))
        pygame.display.flip()

        if not paused:
            frame = (frame + 1) % K

        clock.tick(fps)

    pygame.quit()


# ---------------------------------------------------------------------------
# Frame exporter (for making videos)
# ---------------------------------------------------------------------------

def export_frames(
    pos: NDArray,
    out_dir: str | Path,
    width: int = WINDOW_WIDTH,
    height: int = WINDOW_HEIGHT,
    trail_length: int = 10,
) -> None:
    """Render each frame to a PNG file without displaying a window.

    Args:
        pos:          Shape (K, N, 2).
        out_dir:      Directory to write frame_0000.png ... frame_KKKK.png.
        width:        Image width in pixels.
        height:       Image height in pixels.
        trail_length: Number of trailing frames to draw.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K, N, _ = pos.shape

    pygame.init()
    surface = pygame.Surface((width, height))

    for frame in range(K):
        surface.fill(BG_COLOR)

        # Trails.
        for obj in range(N):
            color = PALETTE[obj % len(PALETTE)]
            t_start = max(0, frame - trail_length)
            for t in range(t_start, frame):
                alpha = int(255 * (t - t_start) / max(trail_length, 1))
                trail_color = tuple(int(c * alpha / 255) for c in color)
                x, y = int(pos[t, obj, 0]), int(pos[t, obj, 1])
                pygame.draw.circle(surface, trail_color, (x, y), max(1, DOT_RADIUS - 2))

        # Current dots.
        for obj in range(N):
            color = PALETTE[obj % len(PALETTE)]
            x, y = int(pos[frame, obj, 0]), int(pos[frame, obj, 1])
            pygame.draw.circle(surface, color, (x, y), DOT_RADIUS)

        path = out_dir / f"frame_{frame:04d}.png"
        pygame.image.save(surface, str(path))

    pygame.quit()
    print(f"Exported {K} frames to {out_dir}/")
    print("Combine with: ffmpeg -r 30 -i frame_%04d.png -pix_fmt yuv420p output.mp4")
