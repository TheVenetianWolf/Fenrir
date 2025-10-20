"""Unit tests for FENRIR core engine/dynamics/guidance.

These tests are intentionally small and deterministic. They don’t touch Streamlit.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.entities import State
from core.engine import Missile, Target, World
from core.dynamics import BasicMissileDynamics, QuadraticDrag
from core.guidance import ProportionalNavigation
from services.scenarios import simple_pn_intercept


# ----------------------------
# Fixtures / helpers
# ----------------------------
@pytest.fixture
def zero_drag() -> QuadraticDrag:
    # Turn drag off to make some assertions sharper
    return QuadraticDrag(Cd=0.0, A=0.0, rho=0.0)


def mk_state(x=0.0, y=0.0, vx=0.0, vy=0.0, m=1.0) -> State:
    return State(t=0.0, r=np.array([x, y], dtype=float), v=np.array([vx, vy], dtype=float), m=m)


class MockGuidance:
    """Guidance that returns a huge a_lat to test saturation & call wiring."""
    def __init__(self, N: float = 3.0):
        self.N = N
        self.calls = 0
        self.last_inputs = None

    def command(self, missile: Missile, target: Target, world: World) -> dict:
        self.calls += 1
        self.last_inputs = (missile, target, world)
        return {"a_lat": 1e6}  # absurd on purpose: should be saturated by engine/dynamics


# ----------------------------
# Tests
# ----------------------------
def test_world_step_advances_time_and_positions(zero_drag: QuadraticDrag):
    """World.step should advance time and move both entities (CV target, controlled missile)."""
    dt = 0.1
    m_state = mk_state(x=0, y=0, vx=300, vy=0, m=50)
    t_state = mk_state(x=2000, y=500, vx=-150, vy=0, m=200)

    guidance = MockGuidance()
    dynamics = BasicMissileDynamics(amax=50.0, drag=zero_drag)
    missile = Missile(state=m_state, guidance=guidance, dynamics=dynamics, amax=50.0)
    target = Target(state=t_state)
    world = World(entities=[missile, target])

    # Step once
    world.step(dt)

    assert math.isclose(world.t, dt, rel_tol=0, abs_tol=1e-12)
    # Target should move along its velocity
    assert target.state.r[0] < 2000  # moved left
    # Missile should have moved forward
    assert missile.state.r[0] > 0

    # Guidance called exactly once
    assert guidance.calls == 1
    # Command saturated at +/- amax
    assert abs(missile.last_a_cmd) <= missile.amax + 1e-9


def test_dynamics_applies_lateral_accel_perpendicular(zero_drag: QuadraticDrag):
    """Positive a_lat with vx>0 should initially produce +vy (left-hand normal)."""
    dt = 0.05
    m_state = mk_state(x=0, y=0, vx=300, vy=0, m=50)
    t_state = mk_state(x=1000, y=0, vx=0, vy=0, m=200)

    # Guidance returns finite positive a_lat
    class PosGuidance:
        def __init__(self): self.N = 3.0
        def command(self, missile, target, world): return {"a_lat": 20.0}

    dynamics = BasicMissileDynamics(amax=50.0, drag=zero_drag)
    missile = Missile(state=m_state, guidance=PosGuidance(), dynamics=dynamics, amax=50.0)
    target = Target(state=t_state)
    world = World(entities=[missile, target])

    world.step(dt)

    # Expect missile vy to become positive (turning "left" from +x heading)
    assert missile.state.v[1] > 0.0


def test_basic_missile_dynamics_respects_amax(zero_drag: QuadraticDrag):
    """Even with absurd guidance commands, applied lateral accel should not exceed amax."""
    dt = 0.01
    m_state = mk_state(vx=300, vy=0, m=50)
    t_state = mk_state(x=500, y=0, vx=0, vy=0, m=200)

    g = MockGuidance()
    dyn = BasicMissileDynamics(amax=25.0, drag=zero_drag)
    m = Missile(state=m_state, guidance=g, dynamics=dyn, amax=25.0)
    t = Target(state=t_state)
    w = World(entities=[m, t])

    w.step(dt)

    # Engine stores the (saturated) command in missile.last_a_cmd
    assert abs(m.last_a_cmd) <= 25.0 + 1e-9
    # Dynamics’ reported achieved accel magnitude should be finite and non-negative
    assert m.last_a_achieved >= 0.0


def test_pn_guidance_closes_range():
    """In the default PN scenario, range should decrease over a few steps."""
    w = simple_pn_intercept()
    m, t = w.entities[0], w.entities[1]

    r0 = float(np.linalg.norm(t.state.r - m.state.r))
    for _ in range(20):
        w.step(0.05)
    r1 = float(np.linalg.norm(t.state.r - m.state.r))

    assert r1 < r0, "Range did not decrease under PN - guidance/engine wiring may be broken."
