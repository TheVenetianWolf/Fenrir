
import numpy as np
from core.entities import State
from core.engine import Missile, Target, World
from core.guidance import ProportionalNavigation
from core.dynamics import BasicMissileDynamics, QuadraticDrag

def simple_pn_intercept():
    # existing version but now pass dynamics to Missile
    missile_state = State(0.0, np.array([0.0, 0.0]), np.array([300.0, 0.0]), m=50.0)
    target_state  = State(0.0, np.array([4000.0, 800.0]), np.array([-150.0, 0.0]), m=200.0)

    dyn = BasicMissileDynamics(amax=50.0, drag=QuadraticDrag(Cd=0.6, A=0.015, rho=1.225))
    missile = Missile(state=missile_state, guidance=ProportionalNavigation(N=3.5), dynamics=dyn, amax=50.0)
    target = Target(state=target_state)

    return World(entities=[missile, target], wind=np.array([0.0, 0.0]))

# Evasive target: sideways sinusoid
def pn_vs_evasive():
    w = simple_pn_intercept()
    target = next(e for e in w.entities if isinstance(e, Target))
    # attach a behavior function onto world for target
    w.target_behaviour = dict(amp=250.0, freq=0.002, base_v=target.state.v.copy())

    def step_target(t, dt):
        A = w.target_behaviour["amp"]
        f = w.target_behaviour["freq"]
        base = w.target_behaviour["base_v"]
        # lateral sinusoid in +y relative to base x-motion
        v_y = A * 2*np.pi*f * np.cos(2*np.pi*f * t)
        target.state.v = np.array([base[0], v_y])

    w.step_target_fn = step_target
    return w