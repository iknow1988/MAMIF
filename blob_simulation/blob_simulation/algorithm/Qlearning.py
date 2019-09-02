import numpy as np
from blob_simulation.blob.Blob import Blob
from blob_simulation.tools.tools import Problem
from blob_simulation.constants import d, PLAYER_N, ENEMY_N, FOOD_N
from .utils import euclidean_distance
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

HM_EPISODES = 1000

MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 300  # how often to play through env visually.

start_q_table = None  # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95


def Qlearning(problem: Problem, q_table: dict, epsilon: float = epsilon, blob_range: int = 9):

    SIZE = len(problem.matrix)
    episode_rewards = []

    for episode in range(HM_EPISODES):
        player = Blob(size=blob_range)
    #     food = Blob()
        foods = [Blob(x, y) for x, y in problem.goal_list]
    #     enemy = Blob()
        if episode % SHOW_EVERY == 0:
            print(f"on #{episode}, epsilon is {epsilon}")
            print(
                f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
        else:
            show = False

        episode_reward = 0
        # Each move
        for i in range(200):

            boundary_list = player.get_observable_boundaries(
                dat=problem.matrix)
            min_index_enemy = np.argmin(
                [euclidean_distance((player.x, player.y), p) for p in boundary_list])

            enemy_position = boundary_list[min_index_enemy]

            enemy = Blob(enemy_position[0], enemy_position[1])

            min_index_food = np.argmin(
                [euclidean_distance((player.x, player.y), p) for p in problem.goal_list])

            food = foods[min_index_food]

            obs = (player-food, player-enemy)
            # print(obs)
            if np.random.random() > epsilon:
                # GET THE ACTION
                # Given the state (encoded in obs) what action is the best.
                action = np.argmax(q_table[obs])
            else:
                action = np.random.randint(0, 4)
            # Take the action!
            # Note that we have changed the x,y of player.
            player.action(action)

            #### MAYBE ###
            # enemy.move()
            # food.move()
            ##############

            if player.x == enemy.x and player.y == enemy.y:
                reward = -ENEMY_PENALTY
            elif player.x == food.x and player.y == food.y:
                reward = FOOD_REWARD
            else:
                reward = -MOVE_PENALTY
            # NOW WE KNOW THE REWARD, LET'S CALC YO
            # first we need to obs immediately after the move.
            new_obs = (player-food, player-enemy)

            max_future_q = np.max(q_table[new_obs])
            current_q = q_table[obs][action]

            if reward == FOOD_REWARD:
                new_q = FOOD_REWARD
            else:
                new_q = (1 - LEARNING_RATE) * current_q + \
                    LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[obs][action] = new_q

            if show:
                # starts an rbg of our size
                env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

                # sets the player tile to blue
                env[player.x][player.y] = d[PLAYER_N]

                max_x = len(problem.matrix[0])
                max_y = len(problem.matrix)

                for i in range(max_y):
                    for j in range(max_x):
                        if problem.matrix[i][j] == 1:
                            #print('{} {}'.format(j , i))
                            env[i][j] = d[ENEMY_N]
                        elif j == max_x - 1:
                            if problem.matrix[i][j] == 0:
                                env[i][j] = d[FOOD_N]

                img = Image.fromarray(env, 'RGB')

                # resizing so we can see our agent in all its glory.
                img = img.resize((500, 500))
                cv2.imshow("image", np.array(img))  # show it!
                # time.sleep(0.05)
                # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                # if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                #     if cv2.waitKey(500) & 0xFF == ord('q'):
                #         break
                # else:
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                cv2.waitKey(1)

            episode_reward += reward
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                break

        # print(episode_reward)
        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

    moving_avg = np.convolve(episode_rewards, np.ones(
        (SHOW_EVERY,))/SHOW_EVERY, mode='valid')

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")
    plt.show()
