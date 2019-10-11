import os.path as path
import random

import pandas as pd

from ai.model import Dqn
from settings.constants import ANGLE_RANGE, GAMMA, N_EPISODE, N_MOVES, SEED
from components.Game import Game

random.seed(SEED)

SHAPE_FILE = 'shape'



def training(model_file: str, output_csv: str, bat_speed: int, num_episodes: int = N_EPISODE,
             moves_per_episode: int = N_MOVES, update_only_better_reward: bool = True,shap_file:str='shape'):
    for i in range(num_episodes):

        print("Episode : {}".format(i))
        # INIT ENV FOR TRAINING

        # 1 define model
        brain = Dqn(5, 2 * ANGLE_RANGE + 1, GAMMA)
        # 2 loading and init the new data frame if needed

        if path.exists(model_file):
            brain.load(model_file)
        if path.exists(output_csv):
            data = pd.read_csv(output_csv)
            experiment_number = data['experiment'].max() + 1
        else:
            columns = ['experiment', 'time', 'speed', 'gamma', 'signal1', 'signal2',
                       'signal3', 'distance_to_goal', 'action', 'orientation', 'reward']
            data = pd.DataFrame(columns=columns)
            experiment_number = 1

        game = Game(model=brain, experiment_number=experiment_number, bat_speed=bat_speed,training_mode=True,
                    shape_file=shap_file)

        for _ in range(moves_per_episode):
            # Each update called would result an additional one row of data store in game.state.sample
            game.update()

        if update_only_better_reward:

            if data.shape[0] == 0:
                current_max_cumulative_reward = -10000000
            else:
                tmp_df = data[data['experiment'] == (experiment_number - 1)]
                current_max_cumulative_reward = tmp_df.reward.sum()
            new_df = pd.DataFrame(game.state.sample)
            sum_reward = new_df['reward'].sum()
            print('Sum reward {}'.format(sum_reward))
            print("Current max {}".format(current_max_cumulative_reward))
            if sum_reward > current_max_cumulative_reward:
                print("Save data")

                data = pd.concat(
                    [data, new_df])
                data.to_csv(output_csv, index=False)

                print('Save model')

                game.state.brain.save(model_file)

        else:
            print("Save data")

            data = pd.concat(
                [data, pd.DataFrame(game.state.sample)])
            data.to_csv(output_csv, index=False)

            print('Save model')

            game.state.brain.save(model_file)


if __name__ == '__main__':
    for speed in range(1,6):
        model_file = "base_brain_" + str(speed) + '.pth'
        output_csv = "obstacle_speed_" + str(speed) + ".csv"

        training(model_file=model_file, output_csv=output_csv, bat_speed=speed,update_only_better_reward=False,
                 shap_file=SHAPE_FILE)
