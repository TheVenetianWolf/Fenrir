"""Core simulation engine for FENRIR.

This module defines:
- `Entity`: minimal base class with a mutable `State`
- `Missile`: guided entity with dynamics + telemetry fields
- `Target`: constant-velocity (by default), can be overridden via a behavior hook
- `World`: container that advances time and updates entities each step

Stepping order (per `World.step`):
1) Optional target behavior hook (`world.step_target_fn(t, dt)`)
2) Target constant-velocity integration
3) Missile guidance -> lateral acceleration command (saturated to `amax`)
4) Missile dynamics integration (applies lateral accel & any drag model)
5) Time update

Notes
-----
- Units: metres [m], seconds [s], velocities [m/s], accelerations [m/s²]
- 2D kinematics (state vectors are shape (2,))
- `wind` is stored on `World`; your dynamics can read it if implemented.
"""

from __future__ import annotations
from typing import List, Optional, Callable, Dict
import numpy as np
from core.entities import State


# ------------------------------------------------------------------------------
# Base entity types
# ------------------------------------------------------------------------------

class Entity:
    """Base simulation entity holding a mutable `State`."""

    def __init__(self, state: State) -> None:
        self.state: State = state


class Missile(Entity):
    """Guided missile with dynamics and a few telemetry fields.

    Attributes
    ----------
    guidance : object
        Strategy with a `.command(missile, target, world) -> dict` method.
        Must return an entry `"a_lat"` (signed lateral accel command) in m/s².
    dynamics : object
        Object with `.step(missile, dt)` that updates `missile.state` per physics.
        The dynamics may populate `missile.last_a_lat` and `missile.last_a_achieved`.
    amax : float
        Max allowed commanded lateral acceleration [m/s²]. Command is saturated here.
    last_a_cmd : float
        Last **saturated** guidance command passed to dynamics [m/s²].
    last_a_lat : float
        Last signed lateral accel magnitude used by dynamics [m/s²] (optional).
    last_a_achieved : float
        Last achieved total accel magnitude from dynamics [m/s²] (optional).
    """

    def __init__(self, state: State, guidance, dynamics, amax: float = 50.0) -> None:
        super().__init__(state)
        self.guidance = guidance
        self.dynamics = dynamics
        self.amax: float = float(amax)

        # Telemetry fields populated during stepping
        self.last_a_cmd: float = 0.0
        self.last_a_lat: float = 0.0
        self.last_a_achieved: float = 0.0


class Target(Entity):
    """Target entity. By default moves with constant velocity each step."""
    pass


# ------------------------------------------------------------------------------
# World container
# ------------------------------------------------------------------------------

class World:
    """Simulation world holding entities, environment, and time.

    Parameters
    ----------
    entities : list[Entity], optional
        Collection of entities (Missile/Target) in the scenario.
    wind : np.ndarray
        Constant wind vector [m/s] in world axes (x, y). Dynamics may use this.
    """

    def __init__(self, entities: Optional[List[Entity]] = None, wind: np.ndarray = np.array([0.0, 0.0])) -> None:
        self.entities: List[Entity] = entities or []
        self.wind: np.ndarray = np.asarray(wind, dtype=float).reshape(2)
        self.t: float = 0.0

        # Optional hooks a scenario may attach:
        # - step_target_fn(t: float, dt: float) -> None
        #   (see services.scenarios.pn_vs_evasive)
        # - other future per-step hooks can be added similarly.

    # -- convenience finders ----------------------------------------------------
    def get_missile(self) -> Optional[Missile]:
        return next((e for e in self.entities if isinstance(e, Missile)), None)

    def get_target(self) -> Optional[Target]:
        return next((e for e in self.entities if isinstance(e, Target)), None)

    def add_entity(self, e: Entity) -> None:
        self.entities.append(e)

    # -- main integration step --------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance the world by `dt` seconds.

        Stepping order:
        1) Run optional target behavior hook (if present).
        2) Integrate target with constant velocity.
        3) Query guidance for missile lateral accel command and saturate to `amax`.
        4) Let dynamics integrate the missile.
        5) Advance world time.

        Args
        ----
        dt : float
            Time step [s]. Must be positive and finite.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("World.step: dt must be a positive, finite number")

        missile = self.get_missile()
        target = self.get_target()

        # (1) Scenario-provided target behavior (e.g., weaving), if any
        if hasattr(self, "step_target_fn") and callable(getattr(self, "step_target_fn")) and target is not None:
            # Signature: step_target_fn(t, dt)
            self.step_target_fn(self.t, dt)

        # (2) Default target kinematics: constant velocity
        if target is not None:
            target.state.r = target.state.r + target.state.v * dt
            target.state.t = float(target.state.t + dt)

        # (3) Guidance -> lateral acceleration command (scalar)
        if missile is not None and target is not None:
            cmd: Dict[str, float] = missile.guidance.command(missile, target, self) or {}
            a_lat_cmd = float(cmd.get("a_lat", 0.0))

            # Saturate command to missile.amax here (dynamics may also enforce)
            a_lat_cmd = float(np.clip(a_lat_cmd, -missile.amax, missile.amax))
            missile.last_a_cmd = a_lat_cmd

            # Optional: record lambda_dot / v_closing if guidance provides them
            # (kept as a non-breaking courtesy; UI can read if present)
            if "lambda_dot" in cmd:
                missile.last_lambda_dot = float(cmd["lambda_dot"])  # type: ignore[attr-defined]
            if "v_closing" in cmd:
                missile.last_v_closing = float(cmd["v_closing"])    # type: ignore[attr-defined]

            # (4) Integrate missile via its dynamics model
            # Keep dynamics in sync with missile.amax if it uses an internal copy
            if hasattr(missile.dynamics, "amax"):
                missile.dynamics.amax = missile.amax
            # Give dynamics a chance to read wind/world if it knows how
            if hasattr(missile.dynamics, "world"):
                missile.dynamics.world = self

            missile.dynamics.step(missile, dt)

        # (5) Advance world time
        self.t = float(self.t + dt)
