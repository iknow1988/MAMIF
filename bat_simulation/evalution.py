
import os.path as path
import random

import pandas as pd

from ai.model import Dqn
from constants import ANGLE_RANGE, GAMMA, SEED
from training_game_v3 import Game

random.seed(SEED)

MODLE_FILE = 'base_brain.pth'
OUTPUT_CSV = 'eval.csv'

N_EPISODE = 300
N_MOVES = 1500


if __name__ == '__main__':
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

        done = False
        n_actions = 0
        while not done:
            game.update()
            result_of_moves = game.last_action()
            # print(result_of_moves)
            if result_of_moves == 'GOAL' or result_of_moves == 'HIT_TREE':
                done = True
            n_actions += 1
            if n_actions >= N_MOVES:
                done = True

        print("Save data")
        data = pd.concat(
            [data, pd.DataFrame(game.state.sample)])
        data.to_csv(OUTPUT_CSV, index=False)

        # print('Save model')

        # game.state.brain.save(MODLE_FILE)

        # new_df = pd.DataFrame(game.state.sample)
        # print("Totoal reward : {}".format(np.sum(new_df['reward']))
