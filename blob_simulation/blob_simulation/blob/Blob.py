import numpy as np
from typing import List, Tuple
MAX_OBSERVABLE_DISTANCE = 10


class Blob:
    def __init__(self, size: int, x: int = 0, y: int = 0, distance: int = MAX_OBSERVABLE_DISTANCE):
        #         self.x = np.random.randint(0, SIZE)
        #         self.y = np.random.randint(0, SIZE)
        self.x = x
        self.y = y
        self._size = size
        self._observable_distance = MAX_OBSERVABLE_DISTANCE

    def __str__(self):
        return f"<Blob x:{self.x}, y:{self.y}>"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice: int):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self._move(x=1, y=1)
        elif choice == 1:
            self._move(x=-1, y=-1)
        elif choice == 2:
            self._move(x=-1, y=1)
        elif choice == 3:
            self._move(x=1, y=-1)

    def _move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self._size-1:
            self.x = self._size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self._size-1:
            self.y = self._size-1

    def get_observable_boundaries(self, dat: List[List[int]]) -> List[Tuple[int, int]]:

        boundary_list = []
        for y in range(self.y - self._observable_distance, self.y + self._observable_distance):
            for x in range(self.x - self._observable_distance, self.x + self._observable_distance):

                if self._point_in_grid(x, y) and dat[y][x] == 1:

                    boundary_list.append((x, y))

        return boundary_list

    def _point_in_grid(self, x: int, y: int) -> bool:

        if x >= 0 and x <= self._size-1:
            if y >= 0 and y <= self._size-1:
                return True
        return False
