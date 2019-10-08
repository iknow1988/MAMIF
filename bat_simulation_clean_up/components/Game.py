
import pickle
import random
from copy import deepcopy
from typing import List, Tuple

import numpy as np
from kivy.vector import Vector

from ai.model import Dqn
from settings.constants import (ANGLE_RANGE,  BAT_SPEED,
                       GAME_SIZE, GAMMA, LOAD_SAND,
                       MARGIN_TO_GOAL_X_AXIS, OFFSET,
                       REWARD_BETTER_DISTANCE,
                       REWARD_GOAL, REWARD_HIT_TREE, REWARD_MOVE,
                       REWARD_ON_EDGE, SITE_MARGIN)

from .state import State
from .Bat import Bat
from .ObstacleMaker import ObstacleMaker

class Game:
    """The central logic of training process.
    """

    def __init__(self, model: Dqn, experiment_number: int, bat_speed: int = BAT_SPEED,training_mode:bool=True):
        """

        Args:
            model (Dqn): Model object for the decision making.
            experiment_number (int): Experiment number to write into csv.
        """

        self.height = GAME_SIZE
        self.width = GAME_SIZE
        self.training_mode = training_mode
        self.action2rotation = [
            i for i in range(-ANGLE_RANGE, ANGLE_RANGE + 1, 1)]

        # Initialize the game state
        self.state = State()
        self.state.brain = model
        self.state.experiment = experiment_number
        self.state.sand = np.zeros((self.height, self.width))

        # Setup Bat object
        bat_x = MARGIN_TO_GOAL_X_AXIS
        bat_y = random.randint(
            MARGIN_TO_GOAL_X_AXIS, self.height - MARGIN_TO_GOAL_X_AXIS)

        self.bat = Bat(x=bat_x, y=bat_y, speed=bat_speed)

        self.bat._update_sensor_position()
        self.bat._update_sensor_signals(self.state)

        # Setup Obstacle object
        self.obstacles = ObstacleMaker(self.height, self.width)

    def _reset_goal(self):
        valid_goal = False
        while not valid_goal:
            self.state.goal_y = random.randint(
                MARGIN_TO_GOAL_X_AXIS, self.height - MARGIN_TO_GOAL_X_AXIS)
            if self.state.sand[self.state.goal_x, self.state.goal_y] == 0:
                valid_goal = True
        self.state.goal = Vector(self.state.goal_x, self.state.goal_y)

    def _init_goal(self):
        self.state.goal_x = self.width - MARGIN_TO_GOAL_X_AXIS + 10
        self._reset_goal()

    def _game_init(self):
        """Initialize some variables in the gamestate.
        """
        self._init_goal()

        self.state.last_reward = 0
        self.state.last_distance = 0
        self.state.first_update = False
        self.state.sample = []
        self.state.time = 1
        self.state.longueur = self.width
        self.state.largeur = self.height

    def _bat_on_edge(self) -> bool:

        on_edge = False
        if self.bat.pos.x < SITE_MARGIN:

            self.bat.pos.x = SITE_MARGIN + OFFSET
            on_edge = True
        if self.bat.pos.x > self.width - SITE_MARGIN:

            self.bat.pos.x = self.width - SITE_MARGIN - OFFSET
            on_edge = True
        if self.bat.pos.y < SITE_MARGIN:

            self.bat.pos.y = SITE_MARGIN + OFFSET
            on_edge = True

        if self.bat.pos.y > self.height - SITE_MARGIN:

            self.bat.pos.y = self.height - SITE_MARGIN - OFFSET
            on_edge = True

        return on_edge

    def _compute_reward(self):

        distance = self.bat.pos.distance(self.state.goal)

        # (To discourage traversing outside of forest )

        on_edge = self._bat_on_edge()
        if on_edge:

            last_reward = REWARD_ON_EDGE

        # print("x : {} , y: {}".format(int(self.bat.pos.x), int(self.bat.pos.y)))
        if self.state.sand[int(self.bat.pos.x), int(self.bat.pos.y)] > 0:
            self.bat.velocity = Vector(0.2, 0).rotate(self.bat.angle)
            last_reward = REWARD_HIT_TREE
        elif not on_edge:  # otherwise
            self.bat.velocity = Vector(BAT_SPEED, 0).rotate(self.bat.angle)
            last_reward = REWARD_MOVE
            if distance < self.state.last_distance:
                last_reward = REWARD_BETTER_DISTANCE

        if distance < 20:
            valid_goal = False
            self.state.goal_x = self.width - self.state.goal_x
            self._reset_goal()
            last_reward = REWARD_GOAL

        self.state.last_reward = last_reward
        self.state.last_distance = distance

    def _compute_signal(self) -> List:
        """Compute signal (List of features Bat uses to identify its current condition)
            It is the input data for the model.
        Returns:
            List: list of signals
        """
        xx = self.state.goal_x - self.bat.pos.x
        yy = self.state.goal_y - self.bat.pos.y

        orientation = Vector(*self.bat.velocity).angle((xx, yy)) / 180.

        last_signal = [self.bat.signal1, self.bat.signal2,
                       self.bat.signal3, orientation, -orientation]

        self.state.orientation = orientation
        return last_signal

    def update(self):
        """Update state and training model for each move
        """
        if self.state.first_update:
            self._game_init()

            if LOAD_SAND:
                self.state.sand = self.obstacles.load(self.state.sand)
            print("Sum")
            print(np.sum(self.state.sand))
            print(self.state.sand.shape)

        ###############
        # Compute action

        last_signal = self._compute_signal()

        action, loss = self.state.brain.update(
            self.state.last_reward, last_signal)

        rotation = self.action2rotation[action]
        if not self.training_mode:
            self.bat.move(rotation, self.state,compute_obs = True)
        else:
            self.bat.move(rotation, self.state, compute_obs=False)

        self._compute_reward()

        action_result = self.last_action()

        results_row =  {'experiment': self.state.experiment, 'time': self.state.time, 'speed': BAT_SPEED, 'gamma': GAMMA,
                'signal1': self.bat.signal1, 'signal2': self.bat.signal2, 'signal3': self.bat.signal3,
                'distance_to_goal':  self.state.last_distance, 'action': rotation, 'orientation': self.state.orientation,
                'reward':  self.state.last_reward, "loss": loss, "action_result": action_result}
        if not self.training_mode:
            for i in range(-ANGLE_RANGE,ANGLE_RANGE + 1):
                results_row['angle_'+str(i)] = self.bat.observations[i + ANGLE_RANGE]

        self.state.sample.append(results_row)
        self.state.time += 1

    def last_action(self) -> str:
        if self.state.last_reward == REWARD_BETTER_DISTANCE:
            return 'FORWARD'
        elif self.state.last_reward == REWARD_GOAL:
            return "GOAL"
        elif self.state.last_reward == REWARD_HIT_TREE:
            return 'HIT_TREE'
        elif self.state.last_reward == REWARD_MOVE:
            return 'MOVE'
        elif self.state.last_reward == REWARD_ON_EDGE:
            return 'ON_EDGE'
