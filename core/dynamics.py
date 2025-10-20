
import numpy as np

class QuadraticDrag:
    def __init__(self, Cd=0.6, A=0.015, rho=1.225):
        self.Cd, self.A, self.rho = Cd, A, rho

    def force(self, v_air: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(v_air) + 1e-12
        return -0.5 * self.rho * self.Cd * self.A * speed * v_air  # Fd = -1/2 rho Cd A |v| v

class BasicMissileDynamics:
    """Point mass in 2D, no thrust for now (coast). Uses lateral accel command."""
    def __init__(self, amax=50.0, drag: QuadraticDrag | None = None, wind=np.array([0.0, 0.0])):
        self.amax = amax
        self.drag = drag or QuadraticDrag()
        self.wind = wind

    def step(self, missile, dt: float):
        # Decompose commanded lateral accel into a vector perpendicular to velocity
        v = missile.state.v
        speed = np.linalg.norm(v) + 1e-12
        n_hat = np.array([-v[1], v[0]]) / speed  # left-normal
        a_cmd = missile.last_a_cmd  # scalar lateral command (m/s^2)
        a_cmd = float(np.clip(a_cmd, -self.amax, self.amax))
        a_lat_vec = a_cmd * n_hat

        # Drag
        v_air = v - self.wind
        Fd = self.drag.force(v_air)
        a_drag = Fd / max(missile.state.m, 1e-9)

        # Total acceleration (no thrust yet): lateral + drag + gravity(optional later)
        a_total = a_lat_vec + a_drag  # add gravity as [0, -g] if you go 2.5D

        # Integrate (semi-implicit Euler)
        missile.state.v = v + a_total * dt
        missile.state.r = missile.state.r + missile.state.v * dt

        # expose for telemetry
        missile.last_a_lat = float(np.linalg.norm(a_lat_vec)) * np.sign(a_cmd)
        missile.last_a_achieved = float(np.linalg.norm(a_total))  # magnitude
