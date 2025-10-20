
import numpy as np
from core.entities import State

class Entity:
    def __init__(self, state):
        self.state = state

class Missile(Entity):
    def __init__(self, state, guidance, dynamics, amax=50.0):
        super().__init__(state)
        self.guidance = guidance
        self.dynamics = dynamics
        self.amax = amax
        self.last_a_cmd = 0.0
        self.last_a_lat = 0.0
        self.last_a_achieved = 0.0

class Target(Entity):
    pass

class World:
    def __init__(self, entities=None, wind=np.array([0.0, 0.0])):
        self.entities = entities or []
        self.wind = wind
        self.t = 0.0

    def step(self, dt):
        missile = next((e for e in self.entities if isinstance(e, Missile)), None)
        target  = next((e for e in self.entities if isinstance(e, Target)), None)

        if hasattr(self, "step_target_fn") and target:
            self.step_target_fn(self.t, dt)

        if target:
            # constant-velocity target (can be replaced by a behavior fn)
            target.state.r = target.state.r + target.state.v * dt

        if missile and target:
            # guidance produces lateral accel command (scalar)
            cmd = missile.guidance.command(missile, target, self)
            missile.last_a_cmd = float(np.clip(cmd.get("a_lat", 0.0), -missile.amax, missile.amax))
            # integrate with dynamics
            missile.dynamics.amax = missile.amax
            missile.dynamics.step(missile, dt)

        self.t += dt
