
import random
from typing import List

import numpy as np
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.properties import (NumericProperty, ObjectProperty,
                             ReferenceListProperty)
from kivy.uix.widget import Widget
from kivy.vector import Vector

from ai.model import Dqn
from constants import (BAT_SPEED, GAMMA, MARGIN_NO_OBSTICLE, NUM_OBSTABLES,
                       PRINT_PATH, REWARD_BETTER_DISTANCE, REWARD_GOAL,
                       REWARD_HIT_TREE, REWARD_MOVE, REWARD_ON_EDGE)
from state import State


"""Bat widget
"""


class Bat(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def __init__(self, **kwargs):
        super(Bat, self).__init__()
        self._observable_degree = 20
        self._observable_distance = 50
        self.observations = [
            self._observable_distance for i in range(2 * self._observable_degree + 1)]

    def _find_distance_to_closest_obsticles_along_angle(self, angle: float, state: State) -> int:

        for distance in range(1, self._observable_distance):
            point = Vector(self.pos) + Vector(distance, 0).rotate(angle)
            try:
                if state.sand[round(point[0]), round(point[1])] == 1:
                    # print(point)
                    return distance
            except:
                continue
        return self._observable_distance

    def _update_sensor_position(self):
        self.sensor1 = Vector(10, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(10, 0).rotate((self.angle+30) % 360) + self.pos
        self.sensor3 = Vector(10, 0).rotate((self.angle-30) % 360) + self.pos

    def _update_sensor_signals(self, state):
        self.signal1 = int(np.sum(state.sand[int(self.sensor1_x)-10:int(
            self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.

        self.signal2 = int(np.sum(state.sand[int(self.sensor2_x)-10:int(
            self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(state.sand[int(self.sensor3_x)-10:int(
            self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

    def move(self, rotation: float, state: State):
        """Move indirection according to rotation.
        """
        # print("In move {}".format(type(self.signal1)))
        # 1 . UPDATE POSITION, ROTATION AND ANGLE
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        # 2. UPDATE SENSOR POSITIONS.
        self._update_sensor_position()

        # 3. COMPUTE THE SIGNAL VALUES.
        self._update_sensor_signals(state)

        # ADJUST BACK THE POSITION OF SENSORS.
        if self.sensor1_x > state.longueur-10 or self.sensor1_x < 10 or self.sensor1_y > state.largeur-10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > state.longueur-10 or self.sensor2_x < 10 or self.sensor2_y > state.largeur-10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > state.longueur-10 or self.sensor3_x < 10 or self.sensor3_y > state.largeur-10 or self.sensor3_y < 10:
            self.signal3 = 1.

        # 4. COMPUTE THE DISTANCE TO CLOSEST OBSTICLES FOR EACH OBSERVABLE DEGREE
        end_angle = self.angle + self._observable_degree
        start_angle = self.angle - self._observable_degree

        step_size = 1
        # print([i for i in range(start_angle, end_angle + 1, step_size)])
        self.observations = [self._find_distance_to_closest_obsticles_along_angle(
            degree, state) for degree in range(start_angle, end_angle + 1, step_size)]

        # print(len(self.observations))
        # print(self.observations)
        # print(sand)
        # self.canvas.add(Color(1.0, 1.0, 1.0))
        # self.canvas.add(Ellipse(pos=(self.x, self.y), size=(1, 1)))


class Goal(Widget):
    pass


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


class ObstacleWidget(Widget):

    def __init__(self, width, height, **kwargs):
        # make sure we aren't overriding any important functionality
        super(ObstacleWidget, self).__init__(**kwargs)
        # self.sand = np.zeros((width, height))
        # self.sand = np.random.randint(0, 2, size=(width, height))

        self.width = width
        self.height = height
        self.sand = np.zeros((width, height))

    def load(self):
        rectangles = []
        for _ in range(NUM_OBSTABLES):
            pos_x = random.randint(
                MARGIN_NO_OBSTICLE, self.width - MARGIN_NO_OBSTICLE)
            pos_y = random.randint(
                MARGIN_NO_OBSTICLE, self.height - MARGIN_NO_OBSTICLE)
            width = random.randint(10, 40)
            self.sand[pos_x: pos_x + width, pos_y: pos_y + width] = 1
            rectangles.append([pos_x, pos_y, width])
            #
        self.canvas.add(Color(0.8, 0.7, 0))
        for rect in rectangles:
            pos_x = rect[0]
            pos_y = rect[1]
            width = rect[2]
            self.canvas.add(Rectangle(pos=(pos_x, pos_y), size=(width, width)))

    def get_sand(self):
        return self.sand

    def set_size(self, width, height):
        self.width = width
        self.height = height
        self.sand = np.zeros((width, height))
        # self.sand = np.random.randint(0, 1, size=(width, height))


class Game(Widget):

    bat = ObjectProperty(None)
    goal = ObjectProperty(None)

    def __init__(self, **kwargs):

        super(Game, self).__init__()
        self.height = 500
        self.width = 500
        self.action2rotation = [i for i in range(-20, 21, 1)]

        self.state = State()
        self.state.brain = Dqn(5, 41, GAMMA)
        self.state.experiment = 1
        print("EXP : {}".format(self.state.experiment))
        # self.bat = ObjectProperty(None)
        # self.goal = ObjectProperty(None)

    def serve_bat(self):
        self.bat.center = self.center
        self.bat.velocity = Vector(BAT_SPEED, 0)

    def _game_init(self):
        """Initial some variables in the gamestate.
        """
        self.state.goal_x = 10
        self.state.goal_y = self.state.largeur - 10
        self.state.goals_y = [i for i in range(self.state.largeur - 10)]
        self.state.last_reward = 0
        self.state.last_distance = 0
        self.state.first_update = False
        self.state.sample = []
        self.state.time = 1

    def update(self, obstacles: ObstacleWidget, dt):

            # print("In update {}".format(type(self.bat.signal1)))

        self.state.longueur = self.width
        self.state.largeur = self.height
        # longueur = 500
        # largeur = 500
        if self.state.first_update:
            self._game_init()

            # Here we overwrite the height and width in the obsticle object.
            obstacles.set_size(self.state.longueur + 1, self.state.largeur + 1)
            obstacles.load()
            self.state.sand = obstacles.get_sand()
            # print("Sum")
            # print(np.sum(self.state.sand))
            # print(self.state.sand.shape)

        # goal_y = min(goals_y, key=lambda x: np.sqrt(
        #     (self.bat.x - goal_x)**2 + (self.bat.y - x)**2))
        xx = self.state.goal_x - self.bat.x
        yy = self.state.goal_y - self.bat.y

        orientation = Vector(*self.bat.velocity).angle((xx, yy))/180.

        # last_signal = [self.bat.signal1, self.bat.signal2,
        #                self.bat.signal3, orientation, -orientation, *self.bat.observations]
        last_signal = [self.bat.signal1, self.bat.signal2,
                       self.bat.signal3, orientation, -orientation]

        action = self.state.brain.update(self.state.last_reward, last_signal)
        rotation = self.action2rotation[action]
        self.bat.move(rotation, self.state)
        distance = np.sqrt((self.bat.x - self.state.goal_x) **
                           2 + (self.bat.y - self.state.goal_y)**2)
        self.goal.pos = (self.state.goal_x, self.state.goal_y)

        if PRINT_PATH:
            self.canvas.add(Color(135, 206, 235))
            self.canvas.add(
                Ellipse(pos=(self.bat.x, self.bat.y), size=(2, 2)))

        # print("X : {} , Y : {}".format(self.bat.x, self.bat.y))
        # print(self.state.sand.shape)

        if self.state.sand[int(self.bat.x), int(self.bat.y)] > 0:
            self.bat.velocity = Vector(0.2, 0).rotate(self.bat.angle)
            self.canvas.add(Color(255, 0, 0))
            self.canvas.add(Ellipse(pos=(self.bat.x, self.bat.y), size=(2, 2)))
            last_reward = REWARD_HIT_TREE
        else:  # otherwise
            self.bat.velocity = Vector(BAT_SPEED, 0).rotate(self.bat.angle)
            last_reward = REWARD_MOVE
            if distance < self.state.last_distance:
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
            self.state.goal_x = self.width-self.state.goal_x
            self.state.goal_y = np.random.randint(0, self.height)
            last_reward = REWARD_GOAL

        last_distance = distance
        self.state.last_reward = last_reward
        self.state.last_distance = last_distance
        self.state.sample.append({'experiment': self.state.experiment, 'time': self.state.time, 'speed': BAT_SPEED, 'gamma': GAMMA,
                                  'signal1': self.bat.signal1, 'signal2': self.bat.signal2, 'signal3': self.bat.signal3,
                                  'distance_to_goal': last_distance, 'action': rotation, 'orientation': orientation,
                                  'reward': last_reward})
        self.state.time += 1
