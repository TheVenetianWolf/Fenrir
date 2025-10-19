
import numpy as np
from core.entities import State
from core.engine import Missile, Target, World
from core.guidance import ProportionalNavigation

def simple_pn_intercept():
    missile_state = State(t=0.0, r=np.array([0.0, 0.0]), v=np.array([300.0, 0.0]))
    target_state  = State(t=0.0, r=np.array([4000.0, 800.0]), v=np.array([-150.0, 0.0]))
    missile = Missile(state=missile_state, guidance=ProportionalNavigation(N=3.0), amax=50.0)
    target = Target(state=target_state)
    return World(entities=[missile, target])
