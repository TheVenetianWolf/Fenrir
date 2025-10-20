"""Dynamics models for FENRIR.

Includes:
- QuadraticDrag:   F_d = -1/2 * rho * C_d * A * |v_air| * v_air
- BasicMissileDynamics: 2D point-mass, coast (no thrust), lateral accel command

Conventions:
- 2D coordinates in metres [m], seconds [s]
- Velocity [m/s], acceleration [m/s²]
- Wind is world-frame (x, y); dynamics use `v_air = v - wind`
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class QuadraticDrag:
    """Simple quadratic drag model: F_d = -1/2 rho C_d A |v| v."""

    def __init__(self, Cd: float = 0.6, A: float = 0.015, rho: float = 1.225) -> None:
        self.Cd = float(Cd)
        self.A = float(A)
        self.rho = float(rho)

    def force(self, v_air: np.ndarray) -> np.ndarray:
        """Return drag force vector [N] for an air-relative velocity.

        Args:
            v_air: Air-relative velocity [m/s], shape (2,).

        Returns:
            np.ndarray shape (2,): Drag force [N] (opposes motion).
        """
        v_air = np.asarray(v_air, dtype=float).reshape(2)
        speed = float(np.linalg.norm(v_air))
        # Drag proportional to |v| * v; guard against exact zero
        return -0.5 * self.rho * self.Cd * self.A * (speed + 1e-12) * v_air


class BasicMissileDynamics:
    """2D point-mass kinematics for a guided missile (coast, no thrust).

    Applies a **lateral acceleration command** perpendicular to velocity plus drag.
    Gravity is optional (off by default for a pure-2D “flat Earth” model).

    Args:
        amax: Max allowed lateral acceleration magnitude [m/s²].
        drag: Drag model; defaults to `QuadraticDrag()`.
        wind: Constant wind vector [m/s] (x, y). If `world.wind` exists at runtime,
              it overrides this local value each step.
        gravity: Optional gravity acceleration [m/s²] (positive scalar).
                 If provided, acceleration [0, -gravity] is added each step.

    Notes:
        - Integration is semi-implicit (a.k.a. symplectic) Euler:
            v_{k+1} = v_k + a_k * dt
            r_{k+1} = r_k + v_{k+1} * dt
        - Lateral direction is the **left-hand normal** to velocity:
            n_hat = [-v_y, v_x] / ||v||
          If ||v|| ~ 0, we fall back to the previous direction or x-hat.
    """

    def __init__(
        self,
        amax: float = 50.0,
        drag: Optional[QuadraticDrag] = None,
        wind: np.ndarray = np.array([0.0, 0.0]),
        gravity: Optional[float] = None,
    ) -> None:
        self.amax: float = float(amax)
        self.drag: QuadraticDrag = drag if drag is not None else QuadraticDrag()
        self.wind: np.ndarray = np.asarray(wind, dtype=float).reshape(2)
        self.gravity: Optional[float] = float(gravity) if gravity is not None else None

        # Internal cached unit-left-normal; helps if speed ~ 0
        self._last_n_hat: np.ndarray = np.array([0.0, 1.0], dtype=float)

        # Optional back-reference; `World.step` may set this.
        self.world = None  # type: ignore[assignment]

    def _left_normal(self, v: np.ndarray) -> np.ndarray:
        """Compute left-hand unit normal to velocity; robust to ||v|| ~ 0."""
        speed = float(np.linalg.norm(v))
        if speed < 1e-9:
            # fallback to last normal; if still zero, default to +y
            n_hat = self._last_n_hat
            if not np.isfinite(n_hat).all() or np.linalg.norm(n_hat) < 1e-9:
                n_hat = np.array([0.0, 1.0], dtype=float)
            return n_hat
        n_hat = np.array([-v[1], v[0]], dtype=float) / speed
        self._last_n_hat = n_hat
        return n_hat

    def step(self, missile, dt: float) -> None:
        """Advance the missile state by one time step.

        Args:
            missile: Object with `.state.r`, `.state.v`, `.state.m`,
                     and telemetry fields `.last_a_cmd`, `.last_a_lat`, `.last_a_achieved`.
            dt: Time step [s] (must be positive and finite).
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("BasicMissileDynamics.step: dt must be a positive, finite number")

        # Current kinematics
        r = missile.state.r
        v = missile.state.v
        m = float(max(missile.state.m, 1e-9))  # guard mass

        # Lateral commanded accel (saturated here as redundancy; engine also clamps)
        a_cmd = float(np.clip(missile.last_a_cmd, -self.amax, self.amax))
        n_hat = self._left_normal(v)
        a_lat_vec = a_cmd * n_hat  # lateral vector [m/s²]

        # Environment: prefer world.wind if available
        wind_vec = self.wind
        if getattr(self, "world", None) is not None and hasattr(self.world, "wind"):
            wind_vec = np.asarray(self.world.wind, dtype=float).reshape(2)

        # Drag force -> accel
        v_air = v - wind_vec
        Fd = self.drag.force(v_air)             # [N]
        a_drag = Fd / m                         # [m/s²]

        # Gravity (optional 2D “flat” term)
        a_g = np.array([0.0, 0.0], dtype=float)
        if self.gravity is not None and np.isfinite(self.gravity):
            a_g[1] = -abs(self.gravity)

        # Total accel
        a_total = a_lat_vec + a_drag + a_g

        # Semi-implicit Euler
        v_new = v + a_total * dt
        r_new = r + v_new * dt

        # Commit
        missile.state.v = v_new
        missile.state.r = r_new

        # Telemetry exposure
        # Signed lateral magnitude (sign = sign of commanded scalar)
        missile.last_a_lat = float(np.linalg.norm(a_lat_vec)) * (1.0 if a_cmd >= 0.0 else -1.0)
        missile.last_a_achieved = float(np.linalg.norm(a_total))
