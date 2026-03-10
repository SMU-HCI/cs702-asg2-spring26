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

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS, DOT_RADIUS, BG_COLOR, Hotspot


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

def _draw_hotspots(
    surface: pygame.Surface,
    hotspots: list[Hotspot],
    frame: int,
    font: pygame.font.Font,
) -> None:
    """Draw hotspot markers on the surface.

    Each hotspot is shown as a translucent circle with a label.  The circle
    is highlighted when the current frame is near the hotspot's active time.
    """
    for h in hotspots:
        hx, hy = int(h.x), int(h.y)
        is_converge = h.kind == "converge"
        base_color = (220, 60, 60) if is_converge else (60, 60, 220)
        radius = 20

        # Highlight when the current frame is within ±5 of the hotspot time.
        near = abs(frame - h.time_step) <= 5
        alpha = 180 if near else 80

        # Draw a translucent filled circle via a temporary surface.
        circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            circle_surf, (*base_color, alpha), (radius, radius), radius
        )
        surface.blit(circle_surf, (hx - radius, hy - radius))

        # Outline.
        pygame.draw.circle(surface, base_color, (hx, hy), radius, 2)

        # Label.
        label = "C" if is_converge else "D"
        txt = font.render(label, True, base_color)
        surface.blit(txt, (hx - txt.get_width() // 2, hy - txt.get_height() // 2))


def play(
    pos: NDArray,
    title: str = "Trajectory Animation",
    width: int = WINDOW_WIDTH,
    height: int = WINDOW_HEIGHT,
    fps: int = FPS,
    trail_length: int = 10,
    hotspots: list[Hotspot] | None = None,
) -> None:
    """Play the animation and block until the window is closed.

    Args:
        pos:          Shape (K, N, 2) — output positions (time-major).
        title:        Window title string.
        width:        Window width in pixels.
        height:       Window height in pixels.
        fps:          Playback frame rate.
        trail_length: Number of past frames to draw as a fading trail.
        hotspots:     Optional list of Hotspot objects to display on the canvas.
    """
    K, N, _ = pos.shape
    hotspots = hotspots or []

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

        # Draw hotspot markers (behind the dots).
        if hotspots:
            _draw_hotspots(screen, hotspots, frame, font)

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
    hotspots: list[Hotspot] | None = None,
) -> None:
    """Render each frame to a PNG file without displaying a window.

    Args:
        pos:          Shape (K, N, 2).
        out_dir:      Directory to write frame_0000.png ... frame_KKKK.png.
        width:        Image width in pixels.
        height:       Image height in pixels.
        trail_length: Number of trailing frames to draw.
        hotspots:     Optional list of Hotspot objects to display on each frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hotspots = hotspots or []

    K, N, _ = pos.shape

    pygame.init()
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    font = pygame.font.SysFont(None, 24)

    for frame in range(K):
        surface.fill(BG_COLOR)

        # Hotspot markers.
        if hotspots:
            _draw_hotspots(surface, hotspots, frame, font)

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
