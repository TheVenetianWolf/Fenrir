"""Guidance laws for FENRIR.

Currently includes:
- ProportionalNavigation: classic 2D PN (a_lat = N * Vc * lambda_dot)

Conventions:
- Positions [m], velocities [m/s], time [s], angles [rad]
- 2D plane; LOS rate is the z-component of cross(l̂, v_rel) / |r|
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class GuidanceStrategy(ABC):
    """Abstract base class for missile guidance strategies."""

    @abstractmethod
    def command(self, missile, target, world) -> Dict[str, float]:
        """Compute control commands for the missile.

        Args:
            missile: Entity with `.state.r` (np.ndarray), `.state.v` (np.ndarray).
            target:  Entity with `.state.r` (np.ndarray), `.state.v` (np.ndarray).
            world:   World/context object (may contain time, wind, etc.).

        Returns:
            dict: At minimum contains `"a_lat"` [m/s²] (signed lateral accel command).
        """
        raise NotImplementedError


class ProportionalNavigation(GuidanceStrategy):
    """Classic 2D Proportional Navigation (PN).

    Law:
        a_lat = N * Vc * λ̇

    where:
        N   — navigation constant (dimensionless)
        Vc  — closing speed [m/s] (positive when closing)
        λ̇   — line-of-sight (LOS) angular rate [rad/s]

    This implementation assumes perfect state knowledge (no seeker noise/lag).
    Optional noise and clamping can be enabled via constructor args; defaults
    keep behaviour identical to the original version.

    Args:
        N: Navigation constant (typical 2–5).
        los_rate_noise_std: Gaussian noise std for λ̇ [rad/s]. Default 0.0 (off).
        clamp_a_lat: If provided, clamp |a_lat| to this value [m/s²]. Leave None
            to let the downstream dynamics enforce `amax`.

    Notes:
        - Numerical guard `eps` avoids division by zero when range ~ 0.
        - In 2D, the LOS rate is computed using the z-component of the cross product.
    """

    def __init__(
        self,
        N: float = 3.0,
        los_rate_noise_std: float = 0.0,
        clamp_a_lat: Optional[float] = None,
    ) -> None:
        self.N: float = float(N)
        self.los_rate_noise_std: float = float(los_rate_noise_std)
        self.clamp_a_lat: Optional[float] = clamp_a_lat

    @staticmethod
    def _los_rate_and_closing(r_m: np.ndarray, v_m: np.ndarray, r_t: np.ndarray, v_t: np.ndarray) -> tuple[float, float]:
        """Compute (λ̇, Vc) for 2D engagement.

        Args:
            r_m: Missile position [m], shape (2,).
            v_m: Missile velocity [m/s], shape (2,).
            r_t: Target position [m], shape (2,).
            v_t: Target velocity [m/s], shape (2,).

        Returns:
            (lambda_dot [rad/s], v_closing [m/s])
        """
        r = r_t - r_m
        v_rel = v_t - v_m
        rng = float(np.linalg.norm(r))
        eps = 1e-9
        rng_safe = rng + eps

        lhat = r / rng_safe  # unit LOS
        # z-component of cross(lhat, v_rel) divided by range
        lambda_dot = float(np.cross(np.append(lhat, 0.0), np.append(v_rel, 0.0))[2] / rng_safe)
        v_closing = float(-np.dot(v_rel, lhat))
        return lambda_dot, v_closing

    def command(self, missile, target, world) -> Dict[str, float]:
        """Compute a lateral acceleration command using PN.

        Returns:
            dict: {"a_lat": <m/s²>} and extra fields for optional telemetry:
                  "N", "lambda_dot", "v_closing"
        """
        r_m = np.asarray(missile.state.r, dtype=float)
        v_m = np.asarray(missile.state.v, dtype=float)
        r_t = np.asarray(target.state.r, dtype=float)
        v_t = np.asarray(target.state.v, dtype=float)

        lambda_dot, v_closing = self._los_rate_and_closing(r_m, v_m, r_t, v_t)

        # Optional measurement noise on LOS rate (kept off by default)
        if self.los_rate_noise_std > 0.0:
            lambda_dot = float(lambda_dot + np.random.normal(0.0, self.los_rate_noise_std))

        a_cmd = float(self.N * v_closing * lambda_dot)

        # Optional local clamp (usually let dynamics handle amax)
        if self.clamp_a_lat is not None:
            a_cmd = float(np.clip(a_cmd, -self.clamp_a_lat, self.clamp_a_lat))

        # Provide extra fields for downstream telemetry if desired
        return {
            "a_lat": a_cmd,
            "N": float(self.N),
            "lambda_dot": float(lambda_dot),
            "v_closing": float(v_closing),
        }
