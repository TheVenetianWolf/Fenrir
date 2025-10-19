
import numpy as np
from core.entities import State

class Entity:
    def __init__(self, state):
        self.state = state

class Missile(Entity):
    def __init__(self, state, guidance, amax=50.0):
        super().__init__(state)
        self.guidance = guidance
        self.amax = amax

class Target(Entity):
    pass

class World:
    def __init__(self, entities=None, wind=np.array([0.0, 0.0])):
        self.entities = entities or []
        self.wind = wind
        self.t = 0.0

    def step(self, dt):
        # very simple integrator: apply guidance lateral accel to missile as a perpendicular accel
        missile = next((e for e in self.entities if isinstance(e, Missile)), None)
        target = next((e for e in self.entities if isinstance(e, Target)), None)
        if missile and target:
            cmd = missile.guidance.command(missile, target, self)
            a_lat = np.clip(cmd.get("a_lat", 0.0), -missile.amax, missile.amax)
            # Rotate velocity slightly by lateral accel approximation:
            # v_new = v + a_lat * dt * n_hat, where n_hat is perpendicular to velocity
            v = missile.state.v
            speed = np.linalg.norm(v) + 1e-9
            n_hat = np.array([-v[1], v[0]]) / speed
            missile.state.v = missile.state.v + a_lat * dt * n_hat
            missile.state.r = missile.state.r + missile.state.v * dt

        # move target (constant velocity)
        if target:
            target.state.r = target.state.r + target.state.v * dt

        self.t += dt
