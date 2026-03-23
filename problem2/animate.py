"""Standalone PyGame animation viewer that plays from a JSON file.

Reads an animation JSON exported by helper.export_animation_json and
renders it with the same visual style as viewer.py.

Usage::

    python animate.py animation.json
    python animate.py animation.json --fps 15
    python animate.py animation.json --export-frames frames/

Controls:
  SPACE   — pause / resume
  R       — restart from frame 0
  ESC / Q — quit
  LEFT    — step back one frame (while paused)
  RIGHT   — step forward one frame (while paused)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pygame


BG_COLOR = (255, 255, 255)


def load_animation(path: str | Path) -> dict:
    """Load animation data from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    data["positions"] = np.array(data["positions"], dtype=np.float32)
    return data


def _draw_hotspots(
    surface: pygame.Surface,
    hotspots: list[dict],
    frame: int,
    font: pygame.font.Font,
) -> None:
    """Draw hotspot markers on the surface."""
    for h in hotspots:
        hx, hy = int(h["x"]), int(h["y"])
        is_converge = h["kind"] == "converge"
        base_color = (220, 60, 60) if is_converge else (60, 60, 220)
        radius = 20

        near = abs(frame - h["time_step"]) <= 5
        alpha = 180 if near else 80

        circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            circle_surf,
            (*base_color, alpha),
            (radius, radius),
            radius,
        )
        surface.blit(circle_surf, (hx - radius, hy - radius))
        pygame.draw.circle(surface, base_color, (hx, hy), radius, 2)

        label = "C" if is_converge else "D"
        txt = font.render(label, True, base_color)
        surface.blit(txt, (hx - txt.get_width() // 2, hy - txt.get_height() // 2))


def play(data: dict, fps_override: int | None = None) -> None:
    """Play the animation interactively."""
    pos = data["positions"]
    K, N, _ = pos.shape
    width = data.get("width", 800)
    height = data.get("height", 600)
    fps = fps_override or data.get("fps", 30)
    dot_radius = data.get("dot_radius", 4)
    trail_length = data.get("trail_length", 10)
    title = data.get("title", "Trajectory Animation")
    colors = data.get("colors", [])
    hotspots = data.get("hotspots", [])

    # Fallback colours.
    default_palette = [
        (231, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (155, 89, 182),
        (243, 156, 18),
        (26, 188, 156),
        (230, 126, 34),
        (52, 73, 94),
    ]
    if not colors:
        colors = [default_palette[i % len(default_palette)] for i in range(N)]

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

        screen.fill(BG_COLOR)

        if hotspots:
            _draw_hotspots(screen, hotspots, frame, font)

        # Draw trails.
        for obj in range(N):
            color = tuple(colors[obj])
            t_start = max(0, frame - trail_length)
            for t in range(t_start, frame):
                alpha = int(255 * (t - t_start) / max(trail_length, 1))
                trail_color = tuple(int(c * alpha / 255) for c in color)
                x, y = int(pos[t, obj, 0]), int(pos[t, obj, 1])
                pygame.draw.circle(
                    screen,
                    trail_color,
                    (x, y),
                    max(1, dot_radius - 2),
                )

        # Draw current dots.
        for obj in range(N):
            color = tuple(colors[obj])
            x, y = int(pos[frame, obj, 0]), int(pos[frame, obj, 1])
            pygame.draw.circle(screen, color, (x, y), dot_radius)

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


def export_frames(data: dict, out_dir: str | Path) -> None:
    """Render each frame to a PNG file (for making videos with ffmpeg)."""
    pos = data["positions"]
    K, N, _ = pos.shape
    width = data.get("width", 800)
    height = data.get("height", 600)
    dot_radius = data.get("dot_radius", 4)
    trail_length = data.get("trail_length", 10)
    colors = data.get("colors", [])
    hotspots = data.get("hotspots", [])

    default_palette = [
        (231, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (155, 89, 182),
    ]
    if not colors:
        colors = [default_palette[i % len(default_palette)] for i in range(N)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pygame.init()
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    font = pygame.font.SysFont(None, 24)

    for frame in range(K):
        surface.fill(BG_COLOR)

        if hotspots:
            _draw_hotspots(surface, hotspots, frame, font)

        for obj in range(N):
            color = tuple(colors[obj])
            t_start = max(0, frame - trail_length)
            for t in range(t_start, frame):
                alpha = int(255 * (t - t_start) / max(trail_length, 1))
                trail_color = tuple(int(c * alpha / 255) for c in color)
                x, y = int(pos[t, obj, 0]), int(pos[t, obj, 1])
                pygame.draw.circle(
                    surface,
                    trail_color,
                    (x, y),
                    max(1, dot_radius - 2),
                )

        for obj in range(N):
            color = tuple(colors[obj])
            x, y = int(pos[frame, obj, 0]), int(pos[frame, obj, 1])
            pygame.draw.circle(surface, color, (x, y), dot_radius)

        path = out_dir / f"frame_{frame:04d}.png"
        pygame.image.save(surface, str(path))

    pygame.quit()
    print(f"Exported {K} frames to {out_dir}/")
    print("Combine with: ffmpeg -r 30 -i frame_%04d.png -pix_fmt yuv420p output.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play animation from a JSON file")
    parser.add_argument("json_file", type=str, help="Path to animation JSON file")
    parser.add_argument("--fps", type=int, default=None, help="Override playback FPS")
    parser.add_argument(
        "--export-frames",
        type=str,
        default=None,
        help="Export frames as PNGs to this directory instead of playing",
    )
    args = parser.parse_args()

    data = load_animation(args.json_file)

    if args.export_frames:
        export_frames(data, args.export_frames)
    else:
        play(data, fps_override=args.fps)
