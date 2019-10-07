

import numpy as np
from kivy.vector import Vector


from settings.constants import ANGLE_RANGE, BAT_OBSERVABLE_DISTANCE, BAT_SPEED


from .state import State



class Bat:
    """Store bat positions and compute input features for Dqn Model.
    """
    def __init__(self, x: float, y: float, speed: int = BAT_SPEED):

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

