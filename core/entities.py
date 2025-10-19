
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    t: float
    r: np.ndarray   # position vector [x, y]
    v: np.ndarray   # velocity vector [vx, vy]
    m: float = 1.0
