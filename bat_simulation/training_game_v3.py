import pickle
import random
from copy import deepcopy
from typing import List, Tuple

import numpy as np
from kivy.vector import Vector

from ai.model import Dqn
from constants import (ANGLE_RANGE, BAT_OBSERVABLE_DISTANCE, BAT_SPEED,
                       GAME_SIZE, GAMMA, LOAD_SAND, MARGIN_NO_OBSTICLE,
                       MARGIN_TO_GOAL_X_AXIS, NUM_OBSTABLES, OFFSET,
                       PRINT_PATH, RANDOM_OBSTACLES, REWARD_BETTER_DISTANCE,
                       REWARD_GOAL, REWARD_HIT_TREE, REWARD_MOVE,
                       REWARD_ON_EDGE, SHAPE_FILE, SITE_MARGIN)
from state import State


class Bat:
    """Store bat positions and compute input features for Dqn Model.
    """

    def __init__(self, x: float, y: float):

        self.pos = Vector(x, y)
        self._observable_degree = ANGLE_RANGE
        self._observable_distance = BAT_OBSERVABLE_DISTANCE
        self.observations = [
            self._observable_distance for i in range(2 * self._observable_degree + 1)]
        self._distance_to_sensor = 10
        self.velocity = Vector(BAT_SPEED, 0)

        self.signal1: float = None
        self.signal2: float = None
        self.signal3: float = None
        self.sensor1: Vector = None
        self.sensor2: Vector = None
        self.sensor3: Vector = None
        self.angle: float = 0

        self.rotation: float = 0

    def _find_distance_to_closest_obsticles_along_angle(self, angle: float, state: State) -> int:
        """Find the distance to the cloest obsticle along a given angle.

        Args:
            angle (float): Angle of interest
            state (State): Game state

        Returns:
            int: Distance to the cloest obsticle
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
        """ Update sensor postion based on the angle between body sensor

        Args:
            angle (float): Angle between body and sensor

        Returns:
            Vector: Position of the updated sensor
        """

        return Vector(self._distance_to_sensor, 0).rotate(angle) + self.pos

    def _compute_obstacle_density(self, state: State, x: int, y: int, width: int) -> float:
        """Compute obsticle density within a given width centered around (x,y) coord.

        Args:
            state (State): [description]
            x (float): X cord
            y (float): Y cord
            width (float): Size of the range
        Returns:
            float: The density of the obsticle
        """
        max_x, max_y = state.sand.shape
        x = int(x)
        y = int(y)
        max_x = min(max_x, x + width)
        max_y = min(max_y, y + width)
        min_x = max(0, x - width)
        min_y = max(0, y - width)
        n = (max_x - min_x) * (max_y - min_y)
        density = int(np.sum(state.sand[min_x:max_x, min_y:max_y])) / n
        return density

    def _update_sensor_position(self):
        """Update all sensors' position.
           This is called when the body position has changed

        """
        self.sensor1 = self._update_sensor(angle=self.angle)
        self.sensor2 = self._update_sensor(angle=self.angle + 30)
        self.sensor3 = self._update_sensor(angle=self.angle - 30)

    def _update_sensor_signals(self, state):
        """Update the value of all signals.
           This is called when the positions body and sensors have changed

        Args:
            state ([type]): Game state
        """
        self.signal1 = self._compute_obstacle_density(
            state=state, x=int(self.sensor1[0]), y=int(self.sensor1[1]), width=self._observable_distance)

        self.signal2 = self._compute_obstacle_density(
            state=state, x=int(self.sensor2[0]), y=int(self.sensor2[1]), width=self._observable_distance)

        self.signal3 = self._compute_obstacle_density(
            state=state, x=int(self.sensor3[0]), y=int(self.sensor3[1]), width=self._observable_distance)

        max_x, max_y = state.sand.shape
        if self.sensor1[0] > max_x-10 or self.sensor1[0] < 10 or self.sensor1[1] > max_y-10 or self.sensor1[1] < 10:
            self.signal1 = 1.
        if self.sensor2[0] > max_x-10 or self.sensor2[0] < 10 or self.sensor2[1] > max_y-10 or self.sensor2[1] < 10:
            self.signal2 = 1.
        if self.sensor3[0] > max_x-10 or self.sensor3[0] < 10 or self.sensor3[1] > max_y-10 or self.sensor3[1] < 10:
            self.signal3 = 1.

    def move(self, rotation: float, state: State):
        """Move indirection according to rotation.
        """
        # print("In move {}".format(type(self.signal1)))
        # 1 . UPDATE POSITION, ROTATION AND ANGLE
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = (self.angle + self.rotation) % 360

        # 2. UPDATE SENSOR POSITIONS.
        self._update_sensor_position()

        # 3. COMPUTE THE SIGNAL VALUES.
        self._update_sensor_signals(state)

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

        if RANDOM_OBSTACLES:
            for _ in range(NUM_OBSTABLES):
                pos_x = random.randint(
                    MARGIN_NO_OBSTICLE, self.width - MARGIN_NO_OBSTICLE)
                pos_y = random.randint(
                    MARGIN_NO_OBSTICLE, self.height - MARGIN_NO_OBSTICLE)

                width = random.randint(10, 40)
                sand[pos_x: pos_x + width, pos_y: pos_y + width] = 1
        else:
            cells = None
            with open(SHAPE_FILE, 'rb') as f:
                cells = np.array(pickle.load(f))

            max_x, max_y = cells.shape
            print(cells.shape)
            for i in range(max_x):
                for j in range(max_y):
                    if cells[i, max_y-1 - j] > 0:
                        sand[i, j] = 1

        return sand


class Game:
    """The central logic of training process.
    """

    def __init__(self, model: Dqn, experiment_number: int):
        """

        Args:
            model (Dqn): Model object for the decision making.
            experiment_number (int): Experiment number to write into csv.
        """

        self.height = GAME_SIZE
        self.width = GAME_SIZE
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

        self.bat = Bat(x=bat_x, y=bat_y)
        self.bat._update_sensor_position()
        self.bat._update_sensor_signals(self.state)

        # Setup Obstacle object
        self.obstacles = ObstacleMaker(self.height, self.width)

    def _init_goal(self):
        self.state.goal_x = self.width - MARGIN_TO_GOAL_X_AXIS + 10
        valid_goal = False
        while not valid_goal:
            self.state.goal_y = random.randint(
                MARGIN_TO_GOAL_X_AXIS, self.height - MARGIN_TO_GOAL_X_AXIS)
            if self.state.sand[self.state.goal_x, self.state.goal_y] == 0:
                valid_goal = True
        self.state.goal = Vector(self.state.goal_x, self.state.goal_y)

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
            while not valid_goal:
                self.state.goal_y = random.randint(
                    MARGIN_TO_GOAL_X_AXIS, self.height - MARGIN_TO_GOAL_X_AXIS)
                if self.state.sand[self.state.goal_x, self.state.goal_y] == 0:
                    valid_goal = True
            self.state.goal = Vector(self.state.goal_x, self.state.goal_y)
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

        self.bat.move(rotation, self.state)

        self._compute_reward()

        action_result = self.last_action()

        self.state.sample.append(
            {'experiment': self.state.experiment, 'time': self.state.time, 'speed': BAT_SPEED, 'gamma': GAMMA,
                'signal1': self.bat.signal1, 'signal2': self.bat.signal2, 'signal3': self.bat.signal3,
                'distance_to_goal':  self.state.last_distance, 'action': rotation, 'orientation': self.state.orientation,
                'reward':  self.state.last_reward, "loss": loss, "action_result": action_result})
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
