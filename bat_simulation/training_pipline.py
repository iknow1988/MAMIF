
import os.path as path
import random

import pandas as pd

from ai.model import Dqn
from constants import ANGLE_RANGE, GAMMA, N_EPISODE, N_MOVES, SEED
from traing_game_v2 import Game

random.seed(SEED)
N_EPISODE = 40
N_MOVES = 2000

if __name__ == '__main__':
    for speed in range(1, 6):
        MODLE_FILE = "base_brain_"+str(speed) + '.pth'
        OUTPUT_CSV = "obstacle_speed_"+str(speed) + ".csv"

        for i in range(N_EPISODE):

            print("Episode : {}".format(i))
            # INIT ENV FOR TRAINING

            # 1 DEFINE MODEL

            brain = Dqn(5, 2 * ANGLE_RANGE + 1, GAMMA)
            if path.exists(MODLE_FILE):
                brain.load(MODLE_FILE)

            if path.exists(OUTPUT_CSV):

                data = pd.read_csv(OUTPUT_CSV)
                experiment_number = data['experiment'].max() + 1

            else:
                columns = ['experiment', 'time', 'speed', 'gamma', 'signal1', 'signal2',
                           'signal3', 'distance_to_goal', 'action', 'orientation', 'reward']
                data = pd.DataFrame(columns=columns)
                experiment_number = 1

            game = Game(brain, experiment_number)

            for j in range(N_MOVES):
                # Each update called would result an addtional one row of data store in game.state.sample
                game.update()

            print("Save data")
            data = pd.concat(
                [data, pd.DataFrame(game.state.sample)])
            data.to_csv(OUTPUT_CSV, index=False)

            print('Save model')

            game.state.brain.save(MODLE_FILE)
