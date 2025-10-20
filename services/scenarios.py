"""
services/scenarios.py — Scenario definitions for the FENRIR simulation.

Each scenario returns a fully constructed `World` object containing
entities (Missile, Target) and optional environmental effects such as wind
or evasive target motion.

Scenarios serve as “world builders”: they define initial positions,
velocities, and behaviours that can be reused in both the dashboard
and standalone simulations.

Author: Matteo Da Venezia
"""

from __future__ import annotations
import numpy as np

from core.entities import State
from core.engine import Missile, Target, World
from core.guidance import ProportionalNavigation
from core.dynamics import BasicMissileDynamics, QuadraticDrag


# Uncomment these ones only if you want to load YAML scenarios
# import yaml
# from pathlib import Path

def simple_pn_intercept() -> World:
    """
    Construct a basic Proportional Navigation (PN) intercept scenario.

    Missile starts at origin, flying east (+x).
    Target starts ahead and slightly above, flying west (−x).

    Returns
    -------
    World
        A simulation world containing:
        - One missile with PN guidance and simple aerodynamic drag.
        - One constant-velocity target.
        - Zero wind field (still air).
    """
    # --- Initial conditions (2D) ---
    missile_state = State(
        t=0.0,
        r=np.array([0.0, 0.0]),
        v=np.array([300.0, 0.0]),
        m=50.0,
    )
    target_state = State(
        t=0.0,
        r=np.array([4000.0, 800.0]),
        v=np.array([-150.0, 0.0]),
        m=200.0,
    )

    # --- Missile configuration ---
    drag_model = QuadraticDrag(Cd=0.6, A=0.015, rho=1.225)
    dynamics = BasicMissileDynamics(amax=50.0, drag=drag_model)
    guidance = ProportionalNavigation(N=3.5)

    missile = Missile(
        state=missile_state,
        guidance=guidance,
        dynamics=dynamics,
        amax=50.0,
    )
    target = Target(state=target_state)

    # --- Create world ---
    world = World(entities=[missile, target], wind=np.array([0.0, 0.0]))
    return world


def pn_vs_evasive() -> World:
    """
    Construct a PN intercept scenario with a target performing evasive motion.

    The target executes a lateral sinusoidal oscillation in the y-direction,
    modelling a “weaving” aircraft or missile evasion pattern.

    Returns
    -------
    World
        A world identical to `simple_pn_intercept`, but with an attached
        time-varying target velocity function stored as `world.step_target_fn`.
    """
    # Start from the base PN world
    world = simple_pn_intercept()

    # Get the target instance
    target = next(e for e in world.entities if isinstance(e, Target))

    # Define oscillation behaviour parameters
    world.target_behaviour = dict(
        amp=250.0,     # oscillation amplitude (m)
        freq=0.002,    # frequency (Hz)
        base_v=target.state.v.copy(),
    )

    def step_target(t: float, dt: float) -> None:
        """Update target’s lateral velocity component (sinusoidal motion)."""
        A = world.target_behaviour["amp"]
        f = world.target_behaviour["freq"]
        base = world.target_behaviour["base_v"]

        # Lateral sinusoidal velocity (y-direction)
        v_y = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        target.state.v = np.array([base[0], v_y])

    # Attach the function to be called by the engine each step
    world.step_target_fn = step_target

    return world


# -- FOR YAML FILE LOADING SCENARIO -- #
# def load_scenario_from_yaml(path: str) -> World:
#     """
#     Build a World from a YAML preset file.
#     """
#     data = yaml.safe_load(Path(path).read_text())

#     # Parse structure (metadata, missile, target, etc.)
#     m = data["missile"]
#     t = data["target"]
#     w = data["world"]

#     missile_state = State(0.0, np.array(m["position"]), np.array(m["velocity"]), m["mass"])
#     target_state = State(0.0, np.array(t["position"]), np.array(t["velocity"]), t["mass"])

#     drag = QuadraticDrag(**m["dynamics"]["drag"])
#     dyn = BasicMissileDynamics(amax=m["dynamics"]["amax"], drag=drag)
#     missile = Missile(state=missile_state,
#                       guidance=ProportionalNavigation(N=m["guidance"]["N"]),
#                       dynamics=dyn,
#                       amax=m["dynamics"]["amax"])
#     target = Target(state=target_state)
#     return World(entities=[missile, target], wind=np.array(w["wind"]))