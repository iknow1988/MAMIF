"""Bat widget
"""
from typing import List

import numpy as np
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.uix.widget import Widget
from kivy.vector import Vector

from state import State


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
            degree) for degree in range(start_angle, end_angle + 1, step_size)]

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
