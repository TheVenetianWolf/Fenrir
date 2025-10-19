
import numpy as np

class GuidanceStrategy:
    def command(self, missile, target, world):
        raise NotImplementedError

class ProportionalNavigation(GuidanceStrategy):
    def __init__(self, N=3.0):
        self.N = N

    def command(self, missile, target, world):
        r = target.state.r - missile.state.r
        v_rel = target.state.v - missile.state.v
        rng = np.linalg.norm(r) + 1e-9
        los = r / rng
        # scalar LOS rate z-component for 2D: cross(los, v_rel)/rng
        lambda_dot = np.cross(np.append(los, 0.0), np.append(v_rel, 0.0))[2] / rng
        v_closing = -np.dot(v_rel, los)
        a_cmd = self.N * v_closing * lambda_dot
        return {"a_lat": float(a_cmd)}
