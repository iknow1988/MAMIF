import random
from typing import Tuple

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


class Bat:

    def __init__(self, x: float, y: float):

        self.pos = Vector(x, y)
        self._observable_degree = 20
        self._observable_distance = 50
        self.observations = [
            self._observable_distance for i in range(2 * self._observable_degree + 1)]
        self._distance_to_sensor = 10

        self.signal1: float = None
        self.signal2: float = None
        self.signal3: float = None
        self.sensor1: Tuple = None
        self.sensor2: Tuple = None
        self.sensor3: Tuple = None
        self.angle: float = 0
        self.velocity: Vector = None
        self.rotation: float = 0

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

    def _update_sensor(self, angle: float) -> Vector:
        """
        :param angle: Angle between sensor the bat body.
        :return: updated position for a sensor.
        """

        return Vector(self._distance_to_sensor, 0).rotate(angle) + self.pos

    def _compute_obstacle_density(self, state: State, x: int, y: int, width: int) -> float:
        """Compute obsticle density within a given width centered around (x,y) coord.

        Args:
            state (State): [description]
            x (float): [description]
            y (float): [description]
            width (float): [description]

        Returns:
            float: [description]
        """
        max_x, max_y = state.sand.shape
        x = int(x)
        y = int(y)
        max_x = max(max_x, x + width)
        max_y = max(max_y, y + width)
        min_x = max(0, x - width)
        min_y = max(0, y - width)
        n = (max_x - min_x) * (max_y - min_y)
        density = int(np.sum(state.sand[min_x:max_x, min_y:max_y])) / n
        return density

    def _update_sensor_position(self):

        self.sensor1 = self._update_sensor(angle=self.angle)
        self.sensor2 = self._update_sensor(angle=self.angle + 30)
        self.sensor3 = self._update_sensor(angle=self.angle - 30)

    def _update_sensor_signals(self, state):
        self.signal1 = self._compute_obstacle_density(
            state=state, x=int(self.sensor1[0]), y=int(self.sensor1[1]), width=10)

        self.signal2 = self._compute_obstacle_density(
            state=state, x=int(self.sensor2[0]), y=int(self.sensor2[1]), width=10)

        self.signal3 = self._compute_obstacle_density(
            state=state, x=int(self.sensor3[0]), y=int(self.sensor3[1]), width=10)

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
        if self.sensor1[0] > state.longueur - 10 or self.sensor1[0] < 10 or self.sensor1[1] > state.largeur - 10 or self.sensor1[1] < 10:
            self.signal1 = 1.
        if self.sensor2[0] > state.longueur - 10 or self.sensor2[0] < 10 or self.sensor2[1] > state.largeur - 10 or self.sensor2[1] < 10:
            self.signal2 = 1.
        if self.sensor3[0] > state.longueur - 10 or self.sensor3[0] < 10 or self.sensor3[1] > state.largeur - 10 or self.sensor3[1] < 10:
            self.signal3 = 1.

        # 4. COMPUTE THE DISTANCE TO CLOSEST OBSTICLES FOR EACH OBSERVABLE DEGREE
        # end_angle = self.angle + self._observable_degree
        # start_angle = self.angle - self._observable_degree

        # step_size = 1
        # print([i for i in range(start_angle, end_angle + 1, step_size)])
        # self.observations = [self._find_distance_to_closest_obsticles_along_angle(
        #     degree, state) for degree in range(start_angle, end_angle + 1, step_size)]

        # print(len(self.observations))
        # print(self.observations)
        # print(sand)
        # self.canvas.add(Color(1.0, 1.0, 1.0))
        # self.canvas.add(Ellipse(pos=(self.x, self.y), size=(1, 1)))


class ObstacleMaker:
    def __init__(self, height, width):

        self.height = height
        self.width = width

    def load(self, sand: np.array) -> np.array:

        for _ in range(NUM_OBSTABLES):
            pos_x = random.randint(
                MARGIN_NO_OBSTICLE, self.width - MARGIN_NO_OBSTICLE)
            pos_y = random.randint(
                MARGIN_NO_OBSTICLE, self.height - MARGIN_NO_OBSTICLE)

            width = random.randint(10, 40)
            sand[pos_x: pos_x + width, pos_y: pos_y + width] = 1

        return sand


class Game:

    def __init__(self, model, experiment_number):

        self.height = 800
        self.width = 800
        self.action2rotation = [i for i in range(-20, 21, 1)]

        self.state = State()
        self.state.brain = model
        self.state.experiment = experiment_number
        self.state.sand = np.zeros((self.height, self.width))
        self.bat = Bat(x=self.width/2, y=self.height / 2)
        self.bat.velocity = Vector(BAT_SPEED, 0)
        self.bat._update_sensor_position()
        self.bat._update_sensor_signals(self.state)

        self.obstacles = ObstacleMaker(self.height, self.width)

    def _game_init(self):
        """Initialize some variables in the gamestate.
        """
        self.state.goal_x = 10
        self.state.goal_y = np.random.randint(0, self.height)
        self.goal = Vector(self.state.goal_x, self.state.goal_y)
        # self.state.goals_y = [i for i in range(self.state.largeur - 10)]
        self.state.last_reward = 0
        self.state.last_distance = 0
        self.state.first_update = False
        self.state.sample = []
        self.state.time = 1
        self.state.longueur = self.width
        self.state.largeur = self.height

    def update(self):

        if self.state.first_update:
            self._game_init()

            self.state.sand = self.obstacles.load(self.state.sand)
            print("Sum")
            print(np.sum(self.state.sand))
            print(self.state.sand.shape)

        # goal_y = min(goals_y, key=lambda x: np.sqrt(
        #     (self.bat.pos.x - goal_x)**2 + (self.bat.pos.y - x)**2))
        xx = self.state.goal_x - self.bat.pos.x
        yy = self.state.goal_y - self.bat.pos.y

        orientation = Vector(*self.bat.velocity).angle((xx, yy)) / 180.

        # last_signal = [self.bat.signal1, self.bat.signal2,
        #                self.bat.signal3, orientation, -orientation, *self.bat.observations]
        last_signal = [self.bat.signal1, self.bat.signal2,
                       self.bat.signal3, orientation, -orientation]

        action = self.state.brain.update(self.state.last_reward, last_signal)
        rotation = self.action2rotation[action]

        self.bat.move(rotation, self.state)
        distance = np.sqrt((self.bat.pos.x - self.state.goal_x) **
                           2 + (self.bat.pos.y - self.state.goal_y) ** 2)
        self.goal.pos = (self.state.goal_x, self.state.goal_y)

        # print("X : {} , Y : {}".format(self.bat.pos.x, self.bat.pos.y))
        # print(self.state.sand.shape)

        if self.state.sand[int(self.bat.pos.x), int(self.bat.pos.y)] > 0:
            self.bat.velocity = Vector(0.2, 0).rotate(self.bat.angle)

            last_reward = REWARD_HIT_TREE
        else:  # otherwise
            self.bat.velocity = Vector(BAT_SPEED, 0).rotate(self.bat.angle)
            last_reward = REWARD_MOVE
            if distance < self.state.last_distance:
                last_reward = REWARD_BETTER_DISTANCE
        # (To discourage traversing outside of forest )
        if self.bat.pos.x < 10:
            self.bat.pos.x = 10
            last_reward = REWARD_ON_EDGE
        if self.bat.pos.x > self.width - 10:
            self.bat.pos.x = self.width - 10
            last_reward = REWARD_ON_EDGE
        if self.bat.pos.y < 10:
            self.bat.pos.y = 10
            last_reward = REWARD_ON_EDGE
        if self.bat.pos.y > self.height - 10:
            self.bat.pos.y = self.height - 10
            last_reward = REWARD_ON_EDGE

        if distance < 20:
            self.state.goal_x = self.width - self.state.goal_x
            self.state.goal_y = np.random.randint(0, self.height)
            last_reward = REWARD_GOAL

        last_distance = distance
        self.state.last_reward = last_reward
        self.state.last_distance = last_distance
        self.state.sample.append(
            {'experiment': self.state.experiment, 'time': self.state.time, 'speed': BAT_SPEED, 'gamma': GAMMA,
             'signal1': self.bat.signal1, 'signal2': self.bat.signal2, 'signal3': self.bat.signal3,
             'distance_to_goal': last_distance, 'action': rotation, 'orientation': orientation,
             'reward': last_reward})
        self.state.time += 1
