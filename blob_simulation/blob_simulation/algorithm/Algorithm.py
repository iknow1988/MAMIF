from abc import ABCMeta, abstractmethod
from blob_simulation.tools.tools import Problem
from blob_simulation.blob.Blob import Blob
from blob_simulation.constants import d, PLAYER_N, ENEMY_N, FOOD_N
from PIL import Image
import numpy as np
import cv2


algo_params = {'reward': {'MOVE_PENALTY': 1,
                          'ENEMY_PENALTY': 300,
                          'FOOD_REWARD': 25},
               'training': {'epsilon': 0.9,
                            'EPS_DECAY': 0.9998,
                            'LEARNING_RATE': 0.1,
                            'DISCOUNT': 0.95,
                            'N_EPISODES': 1000,
                            'SHOW_EVERY': 200
                            }
               }


class Algorithm(metaclass=ABCMeta):

    def __init__(self, problem: Problem, agent_config: dict = {}, algo_params: dict = algo_params):
        self._problem = problem
        self._agent_config = agent_config
        self._algo_params = algo_params

    @property
    def problem_size(self):
        return len(self._problem.matrix)

    @abstractmethod
    def _pick_action(self) -> int:
        """
        Returns the best action for each move according to the algorithm
        """
        pass

    @abstractmethod
    def training(self):
        pass

    def display(self, player: Blob):
        SIZE = self.problem_size

        # starts an rbg of our size
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

        # sets the player tile to blue
        env[player.x][player.y] = d[PLAYER_N]

        max_x = len(self._problem.matrix[0])
        max_y = len(self._problem.matrix)

        for i in range(max_y):
            for j in range(max_x):
                if self._problem.matrix[i][j] == 1:
                    # print('{} {}'.format(j , i))
                    env[i][j] = d[ENEMY_N]
                elif j == max_x - 1:
                    if self._problem.matrix[i][j] == 0:
                        env[i][j] = d[FOOD_N]

        img = Image.fromarray(env, 'RGB')

        # resizing so we can see our agent in all its glory.
        img = img.resize((500, 500))
        cv2.imshow("image", np.array(img))  # show it!
        # time.sleep(0.05)

        cv2.waitKey(1)
