from enum import IntEnum

import numpy as np


class Action(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3
    confirm = 4

    def as_direction(self) -> np.ndarray:
        match self:
            case Action.left:
                return np.array([-1, 0])
            case Action.right:
                return np.array([1, 0])
            case Action.up:
                return np.array([0, -1])
            case Action.down:
                return np.array([0, 1])
            case Action.confirm:
                return np.array([0, 0])
