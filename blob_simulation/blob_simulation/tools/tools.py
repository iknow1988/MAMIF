import numpy as np
from typing import List, Tuple
from blob_simulation.constants import PLAYER_N, ENEMY_N, FOOD_N, d


class Problem:
    """
    Class to store: Problem information
    """

    def __init__(self, matrix: List[List[int]]):
        self.matrix = matrix

        self.goal_list = self._extract_goal_list()

    def _extract_goal_list(self) -> List[Tuple[int, int]]:

        goal_list = []
        max_x = len(self.matrix[0])
        max_y = len(self.matrix)

        env = np.zeros((max_x, max_y, 3), dtype=np.uint8)

        for i in range(max_y):
            for j in range(max_x):
                if self.matrix[i][j] == 1:

                    env[i][j] = d[ENEMY_N]
                elif j == max_x - 1:
                    if self.matrix[i][j] == 0:
                        env[i][j] = d[FOOD_N]
                        goal_list.append((i, j))
        return goal_list
