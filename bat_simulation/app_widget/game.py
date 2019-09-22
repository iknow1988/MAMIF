
import numpy as np
from kivy.graphics import Color, Ellipse
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.vector import Vector

from ai.model import Dqn
from constants import (BAT_SPEED, GAMMA, PRINT_PATH, REWARD_BETTER_DISTANCE,
                       REWARD_GOAL, REWARD_HIT_TREE, REWARD_MOVE,
                       REWARD_ON_EDGE)
from obstacle import ObstacleWidget
from state import State


class Game(Widget):

    bat = ObjectProperty(None)
    goal = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Game, self).__init__()
        self.height = 500
        self.width = 500
        self.action2rotation = [i for i in range(-20, 21, 1)]
        # self.bat = ObjectProperty(None)
        # self.goal = ObjectProperty(None)

    def serve_bat(self):
        self.bat.center = self.center
        self.bat.velocity = Vector(BAT_SPEED, 0)

    def _game_init(self, state: State):
        """Initial some variables in the gamestate.
        """
        state.goal_x = 10
        state.goal_y = state.largeur - 10
        state.goals_y = [i for i in range(state.largeur - 10)]
        state.last_reward = 0
        state.last_distance = 0
        state.first_update = False

    def update(self, obstacles: ObstacleWidget, state: State, brain: Dqn):

            # print("In update {}".format(type(self.bat.signal1)))

        state.longueur = self.width
        state.largeur = self.height
        # longueur = 500
        # largeur = 500
        if state.first_update:
            self._game_init(state)

            # Here we overwrite the height and width in the obsticle object.
            obstacles.set_size(state.longueur + 1, state.largeur + 1)
            obstacles.load()
            state.sand = obstacles.get_sand()
            print("Sum")
            print(np.sum(state.sand))
            print(state.sand.shape)

        # goal_y = min(goals_y, key=lambda x: np.sqrt(
        #     (self.bat.x - goal_x)**2 + (self.bat.y - x)**2))
        xx = state.goal_x - self.bat.x
        yy = state.goal_y - self.bat.y

        orientation = Vector(*self.bat.velocity).angle((xx, yy))/180.

        # last_signal = [self.bat.signal1, self.bat.signal2,
        #                self.bat.signal3, orientation, -orientation, *self.bat.observations]
        last_signal = [self.bat.signal1, self.bat.signal2,
                       self.bat.signal3, orientation, -orientation]

        action = brain.update(state.last_reward, last_signal)
        rotation = self.action2rotation[action]
        self.bat.move(rotation)
        distance = np.sqrt((self.bat.x - state.goal_x) **
                           2 + (self.bat.y - state.goal_y)**2)
        self.goal.pos = (state.goal_x, state.goal_y)

        if PRINT_PATH:
            self.canvas.add(Color(135, 206, 235))
            self.canvas.add(
                Ellipse(pos=(self.bat.x, self.bat.y), size=(2, 2)))

        # print("X : {} , Y : {}".format(self.bat.x, self.bat.y))
        # print(state.sand.shape)

        if state.sand[int(self.bat.x), int(self.bat.y)] > 0:
            self.bat.velocity = Vector(0.2, 0).rotate(self.bat.angle)
            self.canvas.add(Color(255, 0, 0))
            self.canvas.add(Ellipse(pos=(self.bat.x, self.bat.y), size=(2, 2)))
            last_reward = REWARD_HIT_TREE
        else:  # otherwise
            self.bat.velocity = Vector(BAT_SPEED, 0).rotate(self.bat.angle)
            last_reward = REWARD_MOVE
            if distance < state.last_distance:
                last_reward = REWARD_BETTER_DISTANCE
        # (To discourage traversing outside of forest )
        if self.bat.x < 10:
            self.bat.x = 10
            last_reward = REWARD_ON_EDGE
        if self.bat.x > self.width - 10:
            self.bat.x = self.width - 10
            last_reward = REWARD_ON_EDGE
        if self.bat.y < 10:
            self.bat.y = 10
            last_reward = REWARD_ON_EDGE
        if self.bat.y > self.height - 10:
            self.bat.y = self.height - 10
            last_reward = REWARD_ON_EDGE

        if distance < 20:
            state.goal_x = self.width-state.goal_x
            state.goal_y = np.random.randint(0, self.height)
            last_reward = REWARD_GOAL

        last_distance = distance
        state.sample.append({'experiment': state.experiment, 'time': state.time, 'BAT_SPEED': BAT_SPEED, 'gamma': GAMMA,
                             'signal1': self.bat.signal1, 'signal2': self.bat.signal2, 'signal3': self.bat.signal3,
                             'distance_to_goal': last_distance, 'action': rotation, 'orientation': orientation,
                             'reward': last_reward})
        state.time += 1
