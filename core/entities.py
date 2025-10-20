"""Entity state containers for FENRIR.

Defines a minimal mutable `State` dataclass used by missiles/targets:
- 2D position r = [x, y]  [m]
- 2D velocity v = [vx, vy] [m/s]
- mass m [kg]
- time t [s]

Utilities:
- shape/dtype validation on init
- convenience methods for copying, speed, and CV advancement

Notes:
- The class is intentionally *not* frozen so the engine can update fields in-place.
- Arrays are coerced to float64 and shape (2,) on construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class State:
    """Mutable 2D point-mass state.

    Attributes:
        t: Simulation time [s].
        r: Position vector [m], shape (2,).
        v: Velocity vector [m/s], shape (2,).
        m: Mass [kg] (must be > 0).

    Example:
        >>> s = State(t=0.0, r=np.array([0.0, 0.0]), v=np.array([300.0, 0.0]), m=50.0)
        >>> s.speed()
        300.0
        >>> s.advance_cv(0.1)  # constant-velocity integration
        >>> s.r
        array([30.,  0.])
    """

    t: float
    r: np.ndarray  # position [x, y] in metres
    v: np.ndarray  # velocity [vx, vy] in m/s
    m: float = 1.0

    # --- init-time coercion/validation ----------------------------------------
    def __post_init__(self) -> None:
        # Coerce arrays to float64 and ensure 1D of length 2
        self.r = self._coerce_vec(self.r, name="r")
        self.v = self._coerce_vec(self.v, name="v")

        if not np.isfinite(self.t):
            raise ValueError("State.t must be finite")
        if not np.isfinite(self.m) or self.m <= 0.0:
            raise ValueError("State.m must be a positive, finite mass")

    @staticmethod
    def _coerce_vec(x: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape != (2,):
            raise ValueError(f"State.{name} must be a length-2 vector; got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"State.{name} must be finite; got {arr}")
        return arr

    # --- convenience methods ---------------------------------------------------
    def copy(self) -> "State":
        """Deep copy of the state (arrays are copied)."""
        return State(t=float(self.t), r=self.r.copy(), v=self.v.copy(), m=float(self.m))

    def speed(self) -> float:
        """Return ||v||, the speed [m/s]."""
        return float(np.linalg.norm(self.v))

    def unit_velocity(self, eps: float = 1e-12) -> np.ndarray:
        """Return unit vector along velocity; guarded against near-zero speed.

        Args:
            eps: Small value to avoid division by zero.

        Returns:
            np.ndarray: Unit vector of shape (2,).
        """
        s = self.speed()
        if s < eps:
            return np.array([0.0, 0.0], dtype=float)
        return self.v / s

    def advance_cv(self, dt: float) -> None:
        """Advance position using constant-velocity kinematics.

        Args:
            dt: Time step [s] (must be > 0).
        """
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be a positive, finite number")
        self.r = self.r + self.v * dt
        self.t = float(self.t + dt)

    # --- formatted views -------------------------------------------------------
    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        """Return (t, x, y, vx, vy)."""
        return float(self.t), float(self.r[0]), float(self.r[1]), float(self.v[0]), float(self.v[1])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        t, x, y, vx, vy = self.as_tuple()
        return f"State(t={t:.3f}, r=[{x:.2f}, {y:.2f}], v=[{vx:.2f}, {vy:.2f}], m={self.m:.2f})"
