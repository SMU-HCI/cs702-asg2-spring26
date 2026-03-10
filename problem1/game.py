"""
Problem 1: Dynamical System and Control — Flappy Bird

Coordinate system: y=0 at the bottom of the screen, increasing upward.
Rendering converts to pygame screen coordinates via: screen_y = WINDOW_HEIGHT - y.

Controls (keyboard):
  SPACE   — flap (manual mode) / human flap (human-in-loop mode)
  M       — cycle through modes: manual → pid → mpc → human_in_loop
  R       — reset game
  ESC     — quit
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_WIDTH: int = 800
WINDOW_HEIGHT: int = 500
FPS: int = 60
GRAVITY: float = -80.0  # vertical acceleration (world coords, downward)
FLAP_FORCE: float = 100.0  # upward velocity given per manual flap (world coords)
BG_COLOR = (135, 206, 235)

MODES = ["manual", "pid", "mpc", "human_in_loop"]


# ---------------------------------------------------------------------------
# Bird
# ---------------------------------------------------------------------------
@dataclass
class Bird:
    x: float = 100.0
    y: float = WINDOW_HEIGHT / 2  # world y (0 = bottom)
    vx: float = 100.0  # horizontal speed (pixels/s); increases with score
    vy: float = 0.0  # vertical velocity (world coords; positive = up)
    w: float = 20.0
    h: float = 20.0


def bird_motion(bird: Bird, control: float, dt: float) -> None:
    """Update bird position and velocity.

    Args:
        bird:    Bird state (modified in place).
        control: Vertical acceleration input from the controller (world coords).
        dt:      Time step in seconds.
    """
    bird.vy += (GRAVITY + control) * dt
    bird.y += bird.vy * dt
    # Clamp to floor only; ceiling is handled by the loose out-of-bounds check.
    bird.y = max(0.0, bird.y)


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------
@dataclass
class Pipe:
    x: float = float(WINDOW_WIDTH)
    h: float = 150.0  # height of the bottom pipe section (world y from bottom)
    gap: float = 120.0  # vertical gap between bottom and top pipe sections
    w: float = 60.0


def pipe_motion(pipe: Pipe, bird: Bird, dt: float) -> bool:
    """Move pipe leftward relative to the bird and reset when off-screen.

    Args:
        pipe:  Pipe state (modified in place).
        bird:  Bird state (provides horizontal speed).
        dt:    Time step in seconds.

    Returns:
        True when the pipe resets (bird successfully passed a gap).
    """
    pipe.x -= bird.vx * dt

    if pipe.x + pipe.w < 0:
        pipe.x = float(WINDOW_WIDTH)
        pipe.h = random.uniform(60.0, WINDOW_HEIGHT - pipe.gap - 60.0)
        return True  # bird cleared a pipe

    return False


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------
def check_collision(bird: Bird, pipe: Pipe) -> bool:
    """Return True if the bird overlaps with any part of the pipe."""
    bx1, bx2 = bird.x, bird.x + bird.w
    by1, by2 = bird.y, bird.y + bird.h

    px1, px2 = pipe.x, pipe.x + pipe.w

    if bx2 <= px1 or bx1 >= px2:
        return False  # no horizontal overlap

    gap_bottom = pipe.h
    gap_top = pipe.h + pipe.gap

    # Collision if bird is below the gap bottom or above the gap top
    return by1 < gap_bottom or by2 > gap_top


# ---------------------------------------------------------------------------
# 1.1  PID Controller
# ---------------------------------------------------------------------------
@dataclass
class PIDController:
    Kp: float = 1.5
    Ki: float = 0.001
    Kd: float = 0.5
    error_accumulator: float = 0.0
    prev_error: float = 0.0
    max_accumulator: float = 200.0
    dt: float = 1.0 / 60.0

    def reset(self) -> None:
        """Reset the controller state."""
        self.error_accumulator = 0.0
        self.prev_error = 0.0

    def calc_input(
        self,
        set_point: float,
        process_var: float,
        velocity: float = 0.0,
        umin: float = -500.0,
        umax: float = 500.0,
    ) -> float:
        """Calculate the PID control signal.

        Args:
            set_point:   Target value (desired height).
            process_var: Current measured value (current height).
            velocity:    Current vertical velocity (for derivative/feedforward).
            umin:        Minimum control output.
            umax:        Maximum control output.

        Returns:
            Clamped control signal in [umin, umax].
        """
        error = set_point - process_var

        # TODO (1.1): Implement the PID control algorithm.
        #
        #   Steps:
        #     1. Accumulate error for the integral term:
        #          self.error_accumulator += error * self.dt
        #        Apply anti-windup by clamping to ±self.max_accumulator.
        #
        #     2. Compute derivative term.  Two options:
        #          a) Finite difference:  (error - self.prev_error) / self.dt
        #          b) Velocity feedback:  -velocity   (avoids derivative kick)
        #
        #     3. Combine terms:
        #          u = Kp * error + Ki * error_accumulator + Kd * derivative
        #
        #     4. Clamp output to [umin, umax].
        #
        #     5. Update self.prev_error = error before returning.
        #
        # Adjust the Kp, Ki, Kd parameters to achieve good performance (stable flight,
        # responsive control, minimal overshoot). 
        # Remove or replace the line below once you implement the algorithm.
        return 0.0


# ---------------------------------------------------------------------------
# 1.2  MPC Controller
# ---------------------------------------------------------------------------
class MPCController:
    """Model Predictive Controller for the Flappy Bird game.

    Optimizes a sequence of control inputs over a finite prediction horizon
    by simulating the bird dynamics and minimising a cost function.
    """

    def __init__(
        self,
        horizon: int = 30,
        dt: float = 1.0 / 60.0,
        umin: float = -500.0,
        umax: float = 500.0,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.umin = umin
        self.umax = umax
        # Warm-start: shift the previous solution by one step each call.
        self._last_inputs: list[float] = [0.0] * horizon

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _simulate(
        self,
        y0: float,
        vy0: float,
        inputs: list[float],
    ) -> list[tuple[float, float]]:
        """Simulate bird vertical dynamics over the prediction horizon.

        Args:
            y0:     Initial y position (world coords).
            vy0:    Initial vertical velocity.
            inputs: Control inputs, one per time step.

        Returns:
            List of (y, vy) for each step.
        """
        states: list[tuple[float, float]] = []
        y, vy = y0, vy0
        for u in inputs:
            vy += (GRAVITY + u) * self.dt
            y += vy * self.dt
            y = max(0.0, min(float(WINDOW_HEIGHT), y))
            states.append((y, vy))
        return states

    def _cost(
        self,
        states: list[tuple[float, float]],
        target: float,
        inputs: list[float],
    ) -> float:
        """Evaluate the MPC cost for a candidate input sequence.

        Args:
            states: Simulated (y, vy) pairs over the horizon.
            target: Desired y position (centre of the pipe gap).
            inputs: Corresponding control inputs.

        Returns:
            Scalar cost (lower is better).
        """
        # TODO (1.2): Design a meaningful cost function.
        #
        #   Suggested terms:
        #     - Tracking error:  sum((y - target)**2 for y, _ in states)
        #     - Control effort:  sum(u**2 for u in inputs)  * weight
        #     - Terminal cost:   extra weight on the final state error
        #     - Safety penalty:  large value if y hits 0 or WINDOW_HEIGHT
        #
        # Placeholder: tracking error only (no effort penalty).
        total = sum((y - target) ** 2 for y, _ in states)
        return total

    def _optimize(self, y0: float, vy0: float, target: float) -> list[float]:
        """Find the input sequence that minimises self._cost().

        TODO (1.2): Replace this stub with a real optimisation method.
        """
        # Stub: return zero inputs (bird will fall).
        return [0.0] * self.horizon

    # ------------------------------------------------------------------
    # Public interface (same signature as PIDController.calc_input)
    # ------------------------------------------------------------------
    def calc_input(
        self,
        set_point: float,
        process_var: float,
        velocity: float = 0.0,
        umin: float = -500.0,
        umax: float = 500.0,
    ) -> float:
        """Return the first control action from the optimised sequence.

        Args:
            set_point:   Target y position (gap centre).
            process_var: Current y position.
            velocity:    Current vertical velocity (vy).
            umin:        Minimum control value (passed to optimiser).
            umax:        Maximum control value (passed to optimiser).

        Returns:
            First element of the optimised input sequence.
        """
        # Warm-start: shift previous solution left and pad with zero.
        self._last_inputs = self._last_inputs[1:] + [0.0]

        # TODO (1.2): Call self._optimize and store the result.
        best_inputs = self._optimize(process_var, velocity, set_point)
        self._last_inputs = best_inputs

        return best_inputs[0]


# ---------------------------------------------------------------------------
# Control signal (used by both automated controllers)
# ---------------------------------------------------------------------------
def calculate_control_signal(bird: Bird, pipe: Pipe, controller) -> float:
    """Calculate the control signal for the bird.

    Args:
        bird:       Current bird state.
        pipe:       Current pipe state.
        controller: Controller instance with a ``calc_input`` method.

    Returns:
        Control signal value.
    """
    # Only consider pipes that are ahead of the bird.
    if pipe.x + pipe.w < bird.x:
        return 0.0

    # Target: centre of the gap (world y coords).
    target_height = pipe.h + pipe.gap / 2

    # Anticipate overshoot by adjusting target with current velocity.
    velocity_offset = bird.vy * 0.2
    adjusted_target = target_height - velocity_offset

    current_height = bird.y + bird.h / 2
    distance_to_pipe = pipe.x - (bird.x + bird.w)

    # Scale control aggressiveness by proximity to the pipe.
    if distance_to_pipe <= -1:
        distance_factor = 1.5
    else:
        distance_factor = max(0.5, min(1.5, 1 + 1 / (distance_to_pipe + 1)))

    return (
        controller.calc_input(adjusted_target, current_height, bird.vy)
        * distance_factor
    )


# ---------------------------------------------------------------------------
# 1.3  Human-in-the-Loop control signal
# ---------------------------------------------------------------------------
def calculate_control_signal_human(
    bird: Bird,
    pipe: Pipe,
    controller,
    human_flap: bool,
    alpha: float = 0.5,
) -> float:
    """Blend the human's flap input with an automated controller's output.

    Args:
        bird:        Current bird state.
        pipe:        Current pipe state.
        controller:  Automated controller (PID or MPC).
        human_flap:  True if the human pressed the flap key this frame.
        alpha:       Weight for the human input in [0, 1].
                     alpha=1 → pure human, alpha=0 → pure controller.

    Returns:
        Blended control signal.
    """
    human_signal = FLAP_FORCE if human_flap else 0.0
    auto_signal = calculate_control_signal(bird, pipe, controller)

    # TODO (1.3): Design a blending / filtering strategy.
    #
    #   Ideas to consider:
    #     A) Simple linear blend:
    #          return alpha * human_signal + (1 - alpha) * auto_signal
    #
    #     B) Adaptive alpha — increase controller authority near obstacles:
    #          danger = 1 - clamp(distance_to_pipe / threshold, 0, 1)
    #          effective_alpha = alpha * (1 - danger)
    #
    #     C) Input filtering — suppress human flap if bird is above target:
    #          if human_flap and bird.y + bird.h/2 > target_height + margin:
    #              human_signal = 0
    #
    #     D) Predictive override — use controller when human input would
    #        cause a collision within a short look-ahead window.
    #
    # Placeholder: simple linear blend.
    return alpha * human_signal + (1 - alpha) * auto_signal


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _world_to_screen_y(world_y: float) -> int:
    """Convert world y (0 = bottom) to pygame screen y (0 = top)."""
    return int(WINDOW_HEIGHT - world_y)


def draw_bird(surface: pygame.Surface, bird: Bird) -> None:
    screen_y = _world_to_screen_y(bird.y + bird.h)
    rect = pygame.Rect(int(bird.x), screen_y, int(bird.w), int(bird.h))
    pygame.draw.rect(surface, (0, 200, 0), rect)


def draw_pipe(surface: pygame.Surface, pipe: Pipe) -> None:
    # Bottom pipe: from y=0 (screen bottom) up to pipe.h (world)
    bottom_h = int(pipe.h)
    bottom_rect = pygame.Rect(
        int(pipe.x), WINDOW_HEIGHT - bottom_h, int(pipe.w), bottom_h
    )
    # Top pipe: from world y = pipe.h + pipe.gap to top of window
    top_world_start = pipe.h + pipe.gap
    top_h = int(WINDOW_HEIGHT - top_world_start)
    top_rect = pygame.Rect(int(pipe.x), 0, int(pipe.w), max(0, top_h))
    pygame.draw.rect(surface, (0, 150, 0), bottom_rect)
    pygame.draw.rect(surface, (0, 150, 0), top_rect)


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flappy Bird — CS702 Asg 2")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)

    mode_idx = 0  # start in manual mode; cycle with M key
    mode = MODES[mode_idx]

    pid = PIDController()
    mpc = MPCController()

    def reset() -> tuple[Bird, Pipe, int]:
        return Bird(), Pipe(), 0

    bird, pipe, score = reset()
    human_flap = False
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        human_flap = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    bird, pipe, score = reset()
                    pid.reset()
                elif event.key == pygame.K_m:
                    mode_idx = (mode_idx + 1) % len(MODES)
                    mode = MODES[mode_idx]
                    print(f"[mode] {mode}")
                elif event.key == pygame.K_SPACE:
                    if mode == "manual":
                        bird.vy = (
                            max(0.0, bird.vy) + FLAP_FORCE
                        )  # cancel downward momentum, then flap
                    elif mode == "human_in_loop":
                        human_flap = True

        # --- Compute control ---
        if mode == "manual":
            control = 0.0
        elif mode == "pid":
            control = calculate_control_signal(bird, pipe, pid)
        elif mode == "mpc":
            control = calculate_control_signal(bird, pipe, mpc)
        elif mode == "human_in_loop":
            control = calculate_control_signal_human(bird, pipe, pid, human_flap)
        else:
            control = 0.0

        # --- Update dynamics ---
        bird_motion(bird, control, dt)
        if pipe_motion(pipe, bird, dt):
            score += 1
            bird.vx += 5.0  # speed boost

        # --- Check game-over ---
        out_of_bounds = bird.y <= 0 or bird.y > WINDOW_HEIGHT * 1.5
        if check_collision(bird, pipe) or out_of_bounds:
            print(f"[game over] score={score}  mode={mode}")
            bird, pipe, score = reset()
            pid.reset()

        # --- Draw ---
        screen.fill(BG_COLOR)
        draw_pipe(screen, pipe)
        draw_bird(screen, bird)
        hud = font.render(
            f"Score: {score}    Mode: {mode}    (M=cycle  R=reset  SPACE=flap  ESC=quit)",
            True,
            (0, 0, 0),
        )
        screen.blit(hud, (10, 10))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
