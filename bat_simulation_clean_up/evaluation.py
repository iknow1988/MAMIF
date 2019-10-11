import os.path as path
import random

import pandas as pd
import time
from ai.model import Dqn
from settings.constants import ANGLE_RANGE, GAMMA, SEED,SHAPE_FILE
from components.Game import Game

# MODLE_FILE = 'obstacle_brain_3.pth'
# OUTPUT_CSV = 'eval2.csv'

N_EPISODE = 10
N_MOVES = 1500

ADDING_OBS = False # Adding bat observations


def evaluate(model: str, output_csv: str, bat_speed: int,shape_file:str=SHAPE_FILE):

    for i in range(N_EPISODE):

        random.seed(random.randint(0, 1000))
        print("Episode : {}".format(i))
        print("S : {}".format(bat_speed))

        brain = Dqn(5, 2 * ANGLE_RANGE + 1, GAMMA)
        if path.exists(model):
            brain.load(model_file=model)
        if path.exists(output_csv):
            data = pd.read_csv(output_csv)
            experiment_number = data['experiment'].max() + 1

        else:
            columns = ['experiment', 'time', 'speed', 'gamma', 'signal1', 'signal2',
                       'signal3', 'distance_to_goal', 'action', 'orientation', 'reward']
            if ADDING_OBS:
                for i in range(-ANGLE_RANGE, ANGLE_RANGE + 1):
                    columns.append('angle_' + str(i))
            data = pd.DataFrame(columns=columns)
            experiment_number = 1
        if ADDING_OBS:
            game = Game(model=brain, experiment_number=experiment_number, bat_speed=bat_speed, training_mode=False,
                        shape_file=shape_file)
        else:
            game = Game(model=brain, experiment_number=experiment_number, bat_speed=bat_speed, training_mode=True,
                        shape_file=shape_file)
        print("Size : {}".format(game.state.sand.shape))
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
        data.to_csv(output_csv, index=False)


if __name__ == '__main__':
    shape_file = '101 sharps'
    output_csv = "eval_speed_" + shape_file + ".csv"
    t1 = time.time()

    for speed in range(1,6):
        model_file = "obstacle_brain_" + str(speed) + '.pth'
        evaluate(model=model_file, output_csv=output_csv, bat_speed=speed,shape_file=shape_file)

    print("Time spend {}".format(time.time() - t1))
