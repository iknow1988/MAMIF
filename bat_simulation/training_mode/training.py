
import os.path as path
import random

import pandas as pd

from ..ai.model import Dqn
from ..training_mode.train_game import Game

random.seed(9001)

MODLE_FILE = 'tmp_brain.pth'
OUTPUT_CSV = 'tt.csv'
GAMMA = 0.8
N_EPISODE = 10
N_MOVES = 2000

# class Trainer:
#     def __init__(self,  bat_config: dict):

#         self.bat_config = bat_config
#         self.prev_data: pd.DataFrame
#         self.experiment: int = 0
#         self.game: Game = None

#     def load_history(self, model_file: str = MODLE_FILE, output_csv_file: str = OUTPUT_CSV):
#         if path.exists(output_csv_file):
#             self.prev_data = pd.read_csv(output_csv_file)
#             self.experiment = self.prev_data['experiment'].max() + 1

#         else:
#             columns = ['experiment', 'time', 'speed', 'gamma', 'signal1', 'signal2',
#                        'signal3', 'distance_to_goal', 'action', 'orientation', 'reward']
#             self.prev_data = pd.DataFrame(columns=columns)
#             self.experiment = 1

#     def save_model(self):
#         print("saving brain...")
#         self.game.model.save(MODLE_FILE)

#     def save_data(self):
#         print("saving data...")
#         data = pd.concat(
#             [self.prev_data, pd.DataFrame(self.game.state.sample)])
#         data.to_csv(OUTPUT_FILE, index=False)

#     def train(self, model: Dqn):

#         self.game = Game(model=model)
#         for i in range(10):

#             self.load_history()


if __name__ == '__main__':
    for i in range(N_EPISODE):

        print("Episode : {}".format(i))
        # INIT ENV FOR TRAINING

        # 1 DEFINE MODEL

        brain = Dqn(5, 41, GAMMA)
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
            game.update()

        print("Save data")
        data = pd.concat(
            [data, pd.DataFrame(game.state.sample)])
        data.to_csv(OUTPUT_FILE, index=False)

        print('Save model')
        game.state.brain.save(MODLE_FILE)
