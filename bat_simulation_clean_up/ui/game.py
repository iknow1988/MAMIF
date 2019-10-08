import os.path as path
import pickle
import random

import numpy as np
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.properties import (NumericProperty, ObjectProperty,
                             ReferenceListProperty)
from kivy.uix.widget import Widget
from kivy.vector import Vector

from ai.model import Dqn
from settings.constants import (ANGLE_RANGE, BAT_OBSERVABLE_DISTANCE, BAT_SPEED, GAMMA,
                                LOAD_SAND, MARGIN_NO_OBSTACLE, MARGIN_TO_GOAL_X_AXIS,
                                NUM_OBSTACLES, OFFSET, PRINT_PATH,
                                RANDOM_OBSTACLES, REWARD_BETTER_DISTANCE, REWARD_GOAL,
                                REWARD_HIT_TREE, REWARD_MOVE, REWARD_ON_EDGE,
                                SHAPE_FILE, SITE_MARGIN)

from components.state import State


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
        """
        Args:
            **kwargs:
        """
        super(Bat, self).__init__()
        self._observable_degree = ANGLE_RANGE
        self._observable_distance = BAT_OBSERVABLE_DISTANCE
        self.observations = [
            self._observable_distance for i in range(2 * self._observable_degree + 1)]
        self._distance_to_sensor = 10

    def _find_distance_to_closest_obsticles_along_angle(self, angle: float, state: State) -> int:
        """
        Args:
            angle (float):
            state (State):
        """
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
        Args:
            angle (float):
        """
        return Vector(self._distance_to_sensor, 0).rotate(angle) + self.pos

    def _compute_obstacle_density(self, state: State, x: int, y: int, width: int) -> float:
        """Compute obsticle density within a given width centered around (x,y)
        coord.

        Args:
            state (State): [description]
            x (int): [description]
            y (int): [description]
            width (int): [description]

        Returns:
            float: [description]
        """
        max_x, max_y = state.sand.shape
        x = int(x)
        y = int(y)
        max_x = min(max_x, x + width)
        max_y = min(max_y, y + width)
        min_x = max(0, x - width)
        min_y = max(0, y - width)
        n = (max_x - min_x) * (max_y - min_y)
        count = int(np.sum(state.sand[min_x:max_x, min_y:max_y]))
        # print("Count : {}".format(count))
        # print("N : {}".format(n))
        density = count / n
        return density

    def _update_sensor_position(self):

        self.sensor1 = self._update_sensor(angle=self.angle)
        self.sensor2 = self._update_sensor(angle=self.angle + 30)
        self.sensor3 = self._update_sensor(angle=self.angle - 30)

    def _update_sensor_signals(self, state):
        """
        Args:
            state:
        """
        self.signal1 = self._compute_obstacle_density(
            state=state, x=int(self.sensor1_x), y=int(self.sensor1_y), width=self._observable_distance)

        self.signal2 = self._compute_obstacle_density(
            state=state, x=int(self.sensor2_x), y=int(self.sensor2_y), width=self._observable_distance)

        self.signal3 = self._compute_obstacle_density(
            state=state, x=int(self.sensor3_x), y=int(self.sensor3_y), width=self._observable_distance)

    def move(self, rotation: float, state: State):
        """Move indirection according to rotation.

        Args:
            rotation (float):
            state (State):
        """
        # print("In move {}".format(type(self.signal1)))
        # 1 . UPDATE POSITION, ROTATION AND ANGLE
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = (self.angle + self.rotation) % 360
        # print("Angle {}".format(self.angle
        #                         ))
        # 2. UPDATE SENSOR POSITIONS.
        self._update_sensor_position()

        # 3. COMPUTE THE SIGNAL VALUES.
        self._update_sensor_signals(state)

        # ADJUST BACK THE POSITION OF SENSORS.
        if self.sensor1_x > state.longueur - 10 or self.sensor1_x < 10 or self.sensor1_y > state.largeur - 10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > state.longueur - 10 or self.sensor2_x < 10 or self.sensor2_y > state.largeur - 10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > state.longueur - 10 or self.sensor3_x < 10 or self.sensor3_y > state.largeur - 10 or self.sensor3_y < 10:
            self.signal3 = 1.
        # print("Signals : {}".format(
        #     [self.signal1, self.signal2, self.signal3]))

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
        """
        Args:
            width:
            height:
            **kwargs:
        """
        super(ObstacleWidget, self).__init__(**kwargs)
        # self.sand = np.zeros((width, height))
        # self.sand = np.random.randint(0, 2, size=(width, height))

        self.width = width
        self.height = height
        self.sand = np.zeros((width, height))

    def load(self):

        rectangles = []
        if RANDOM_OBSTACLES:
            for _ in range(NUM_OBSTACLES):
                pos_x = random.randint(
                    MARGIN_NO_OBSTACLE, self.width - MARGIN_NO_OBSTACLE)
                pos_y = random.randint(
                    MARGIN_NO_OBSTACLE, self.height - MARGIN_NO_OBSTACLE)
                width = random.randint(10, 40)
                self.sand[pos_x: pos_x + width, pos_y: pos_y + width] = 1
                rectangles.append([pos_x, pos_y, width])
        else:
            cells = None
            with open(SHAPE_FILE, 'rb') as f:
                cells = np.array(pickle.load(f))

            max_x, max_y = cells.shape
            print(cells.shape)
            for i in range(max_x):
                for j in range(max_y):
                    if cells[i, max_y - 1 - j] > 0:
                        self.sand[i, j] = 1
                        rectangles.append([i, j, 1])
        print(self.sand.shape)
        print(type(self.sand))
        self.canvas.add(Color(0.8, 0.7, 0))

        for rect in rectangles:
            pos_x = rect[0]
            pos_y = rect[1]
            width = rect[2]
            # print("x : {} , y : {}".format(pos_x, pos_y))
            self.canvas.add(Rectangle(pos=(pos_x, pos_y), size=(width, width)))

    def get_sand(self):
        return self.sand

    def set_size(self, width, height):
        """
        Args:
            width:
            height:
        """
        self.width = width
        self.height = height
        self.sand = np.zeros((width, height))
        # self.sand = np.random.randint(0, 1, size=(width, height))


class Game(Widget):
    bat = ObjectProperty(None)
    goal = ObjectProperty(None)

    def __init__(self, model: str, bat_speed: int):

        super(Game, self).__init__()
        self.height = 500
        self.width = 500
        self.action2rotation = [
            i for i in range(-ANGLE_RANGE, ANGLE_RANGE + 1, 1)]

        self.state = State()
        self.state.bat_speed = bat_speed
        self.state.brain = Dqn(5, 2 * ANGLE_RANGE + 1, GAMMA)
        if path.exists(model):
            print("Loading brain")
            self.state.brain.load(model)

        self.state.experiment = 1

    def _init_bat(self):
        bat_x = MARGIN_TO_GOAL_X_AXIS
        bat_y = random.randint(
            MARGIN_TO_GOAL_X_AXIS, self.height - MARGIN_TO_GOAL_X_AXIS)
        self.bat.pos = [bat_x, bat_y]
        self.bat.velocity = Vector(self.state.bat_speed, 0)

    def _reset_goal(self):
        valid_goal = False
        while not valid_goal:
            self.state.goal_y = random.randint(
                MARGIN_TO_GOAL_X_AXIS, self.height - MARGIN_TO_GOAL_X_AXIS)

            if self.state.sand[self.state.goal_x, self.state.goal_y] == 0:
                valid_goal = True

    def _init_goals(self):
        self.state.goal_x = self.width - MARGIN_NO_OBSTACLE

        self._reset_goal()
        print("Sand Shape : {}".format(self.state.sand.shape))


    def _game_init(self):
        """Initialize some variables in the gamestate."""
        self._init_goals()
        self.state.last_reward = 0
        self.state.last_distance = 0
        self.state.first_update = False
        self.state.sample = []
        self.state.time = 1
        self._init_bat()

    def _bat_on_edge(self) -> bool:
        on_edge = False
        if self.bat.x < SITE_MARGIN:
            # if self.bat.x < 0:
            #     self.bat.x = 0
            self.bat.x = SITE_MARGIN + OFFSET
            on_edge = True
        if self.bat.x > self.width - SITE_MARGIN:
            # if self.bat.x > self.width:
            #     self.bat.x = self.width - 1
            self.bat.x = self.width - SITE_MARGIN - OFFSET
            on_edge = True
        if self.bat.y < SITE_MARGIN:
            # if self.bat.y < 0:
            #     self.bat.y = 0
            self.bat.y = SITE_MARGIN + OFFSET
            on_edge = True

        if self.bat.y > self.height - SITE_MARGIN:
            # if self.bat.y > self.height:
            #     self.bat.y = self.height - 1
            self.bat.y = self.height - SITE_MARGIN - OFFSET
            on_edge = True
        # print("X : {} , Y: {}".format(self.bat.x, self.bat.y))
        return on_edge

    def _compute_reward(self):
        #################
        # Compute reward
        distance = np.sqrt((self.bat.x - self.state.goal_x) **
                           2 + (self.bat.y - self.state.goal_y) ** 2)

        # (To discourage traversing outside of forest )
        on_edge = self._bat_on_edge()
        if on_edge:

            last_reward = REWARD_ON_EDGE

        if self.state.sand[int(self.bat.x), int(self.bat.y)] > 0:
            self.bat.velocity = Vector(0.2, 0).rotate(self.bat.angle)
            self.canvas.add(Color(255, 0, 0))
            self.canvas.add(Ellipse(pos=(self.bat.x, self.bat.y), size=(2, 2)))
            last_reward = REWARD_HIT_TREE
        elif not on_edge:  # otherwise
            self.bat.velocity = Vector(self.state.bat_speed, 0).rotate(self.bat.angle)
            last_reward = REWARD_MOVE
        elif distance < self.state.last_distance:
            last_reward = REWARD_BETTER_DISTANCE

        # (To discourage traversing outside of forest )
        # if self.bat.x < 10:
        #     self.bat.x = 10
        #     last_reward = REWARD_ON_EDGE
        # if self.bat.x > self.width - 10:
        #     self.bat.x = self.width - 10
        #     last_reward = REWARD_ON_EDGE
        # if self.bat.y < 10:
        #     self.bat.y = 10
        #     last_reward = REWARD_ON_EDGE
        # if self.bat.y > self.height - 10:
        #     self.bat.y = self.height - 10
        #     last_reward = REWARD_ON_EDGE

        if distance < 20:
            self.state.goal_x = self.width - self.state.goal_x
            # valid_goal = False
            # while not valid_goal:
            #     self.state.goal_y = np.random.randint(0, self.height)
            #     if self.state.sand[self.state.goal_x, self.state.goal_y] == 0:
            #         valid_goal = True
            self._reset_goal()

            last_reward = REWARD_GOAL

        self.state.last_reward = last_reward
        self.state.last_distance = distance

    def update(self, obstacles: ObstacleWidget, dt):


        self.state.longueur = self.width
        self.state.largeur = self.height

        if self.state.first_update:
            obstacles.set_size(self.state.longueur + 1, self.state.largeur + 1)
            if LOAD_SAND:
                obstacles.load()
            self.state.sand = obstacles.get_sand()
            self._game_init()

            # Here we overwrite the height and width in the obsticle object.

            # print("Sum")
            # print(np.sum(self.state.sand))
            # print(self.state.sand.shape)

        # goal_y = min(goals_y, key=lambda x: np.sqrt(
        #     (self.bat.x - goal_x)**2 + (self.bat.y - x)**2))
        xx = self.state.goal_x - self.bat.x
        yy = self.state.goal_y - self.bat.y

        orientation = Vector(*self.bat.velocity).angle((xx, yy)) / 180.

        # last_signal = [self.bat.signal1, self.bat.signal2,
        #                self.bat.signal3, orientation, -orientation, *self.bat.observations]
        last_signal = [self.bat.signal1, self.bat.signal2,
                       self.bat.signal3, orientation, -orientation]

        action, loss = self.state.brain.update(
            self.state.last_reward, last_signal)
        # print("Action : {}".format(action))
        rotation = self.action2rotation[action]
        # print("Rotation : {}".format(rotation))
        self.bat.move(rotation, self.state)

        self.goal.pos = (self.state.goal_x, self.state.goal_y)

        if PRINT_PATH:
            self.canvas.add(Color(135, 206, 235))
            self.canvas.add(
                Ellipse(pos=(self.bat.x, self.bat.y), size=(2, 2)))

        # print("X : {} , Y : {}".format(self.bat.x, self.bat.y))
        # print(self.state.sand.shape)
        self._compute_reward()
        # print(self.state.last_reward)

        self.state.sample.append(
            {'experiment': self.state.experiment, 'time': self.state.time, 'speed': self.state.bat_speed,
             'gamma': GAMMA,
             'signal1': self.bat.signal1, 'signal2': self.bat.signal2, 'signal3': self.bat.signal3,
             'distance_to_goal': self.state.last_distance, 'action': rotation, 'orientation': orientation,
             'reward': self.state.last_reward, 'loss': loss})
        self.state.time += 1
