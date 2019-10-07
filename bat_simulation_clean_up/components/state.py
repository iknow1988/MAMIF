"""State object that tracks the training data.
"""
from typing import List

import numpy as np

from ai.model import Dqn


class State:
    def __init__(self):
        self.sand: np.array
        self.brain: Dqn
        self.last_reward: float
        self.last_distance: float
        self.goal_x: int
        self.goal_y: int
        self.goals_y: int
        self.longueur: int
        self.largeur: int
        self.time: int
        self.first_update = True
        self.experiment: int
        self.time: int
        self.sample: List
        self.orientation: float
        self.bat_speed: int
