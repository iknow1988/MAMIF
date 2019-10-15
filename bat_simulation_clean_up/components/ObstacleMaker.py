import pickle
import random

import numpy as np

from settings.constants import (MARGIN_NO_OBSTACLE,
                                NUM_OBSTACLES, RANDOM_OBSTACLES, SHAPE_FILE)

class ObstacleMaker:
    def __init__(self, height, width):

        self.height = height
        self.width = width

    def load(self, sand: np.array,shape_file:str=SHAPE_FILE) -> np.array:

        if RANDOM_OBSTACLES:
            for _ in range(NUM_OBSTACLES):
                pos_x = random.randint(
                    MARGIN_NO_OBSTACLE, self.width - MARGIN_NO_OBSTACLE)
                pos_y = random.randint(
                    MARGIN_NO_OBSTACLE, self.height - MARGIN_NO_OBSTACLE)

                width = random.randint(5, 10)
                sand[pos_x: pos_x + width, pos_y: pos_y + width] = 1
        else:
            cells = None
            print("Loading shapefile : {}".format(shape_file))
            with open(shape_file, 'rb') as f:
                cells = np.array(pickle.load(f))

            max_x, max_y = cells.shape
            print(cells.shape)
            for i in range(max_x):
                for j in range(max_y):

                    if cells[i, max_y-1 - j] > 0:
                        sand[i, j] = 1
                    # if cells[i, j] > 0:
                    #     sand[i, j] = 1

        return sand
