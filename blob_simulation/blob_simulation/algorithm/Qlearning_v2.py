from blob_simulation.algorithm.Algorithm import Algorithm
from blob_simulation.tools.tools import Problem
from blob_simulation.blob.Blob import Blob
from blob_simulation.algorithm.utils import euclidean_distance
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


algo_params = {'reward': {'MOVE_PENALTY': 1,
                          'ENEMY_PENALTY': 300,
                          'FOOD_REWARD': 25},
               'training': {'epsilon': 0.9,
                            'EPS_DECAY': 0.9998,
                            'LEARNING_RATE': 0.1,
                            'DISCOUNT': 0.95,
                            'N_EPISODES': 4000,
                            'SHOW_EVERY': 200
                            }
               }


class Qlearning(Algorithm):

    def __init__(self, problem,  q_table, agent_config={}, algo_params=algo_params):
        super().__init__(problem, agent_config, algo_params)
        self._q_table = q_table

    def _get_closest_enemy(self, player: Blob) -> Blob:
        """
        Returns the closest observable obstacle.
        """
        boundary_list = player.get_observable_boundaries(
            dat=self._problem.matrix)

        min_index_enemy = np.argmin(
            [euclidean_distance((player.x, player.y), p) for p in boundary_list])

        enemy_position = boundary_list[min_index_enemy]

        enemy = Blob(enemy_position[0], enemy_position[1])

        return enemy

    def _get_closest_food(self, player: Blob, foods: List[Blob]) -> Blob:

        min_index_food = np.argmin(
            [euclidean_distance((player.x, player.y), p) for p in self._problem.goal_list])

        food = foods[min_index_food]

        return food

    def _pick_action(self, player: Blob, closest_food: Blob, closest_enemy: Blob) -> int:

        obs = (player-closest_food, player-closest_enemy)

        if np.random.random() > self._algo_params['training']['epsilon']:
            # GET THE ACTION
            # Given the state (encoded in obs) what action is the best.
            action = np.argmax(self._q_table[obs])
        else:
            action = np.random.randint(0, 4)

        return action

    def _update_q_table(self, player: Blob, closest_food: Blob, closest_enemy: Blob, action: int, obs: Tuple) -> float:

        ENEMY_PENALTY = self._algo_params['reward']['ENEMY_PENALTY']
        FOOD_REWARD = self._algo_params['reward']['FOOD_REWARD']
        MOVE_PENALTY = self._algo_params['reward']['MOVE_PENALTY']
        LEARNING_RATE = self._algo_params['training']['LEARNING_RATE']
        DISCOUNT = self._algo_params['training']['DISCOUNT']

        if player.x == closest_enemy.x and player.y == closest_enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == closest_food.x and player.y == closest_food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-closest_food, player-closest_enemy)

        max_future_q = np.max(self._q_table[new_obs])
        current_q = self._q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        self._q_table[obs][action] = new_q

        return reward

    def training(self):
        SIZE = self.problem_size
        episode_rewards = []

        SHOW_EVERY = self._algo_params['training']['SHOW_EVERY']
        EPS_DECAY = self._algo_params['training']['EPS_DECAY']

        ENEMY_PENALTY = self._algo_params['reward']['ENEMY_PENALTY']
        FOOD_REWARD = self._algo_params['reward']['FOOD_REWARD']
        MOVE_PENALTY = self._algo_params['reward']['MOVE_PENALTY']

        epsilon = self._algo_params['training']['epsilon']
        MAX_MOVES_PER_EPISODE = 200

        for episode in range(self._algo_params['training']['N_EPISODES']):
            player = Blob(size=SIZE)
            foods = [Blob(x, y) for x, y in self._problem.goal_list]

            if episode % SHOW_EVERY == 0:
                print(f"on #{episode}, epsilon is {epsilon}")
                print(
                    f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
                show = True
            else:
                show = False

            episode_reward = 0
            for _ in range(MAX_MOVES_PER_EPISODE):

                closest_enemy = self._get_closest_enemy(player)

                closest_food = self._get_closest_food(player, foods)

                obs = (player-closest_food, player-closest_enemy)
                best_action = self._pick_action(
                    player, closest_food, closest_enemy)

                player.action(best_action)

                reward = self._update_q_table(player, closest_food,
                                              closest_enemy, best_action, obs)

                if show:
                    self.display(player)
                episode_reward += reward

                if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                    break
            episode_rewards.append(episode_reward)
            epsilon *= EPS_DECAY

        moving_avg = np.convolve(episode_rewards, np.ones(
            (SHOW_EVERY,))/SHOW_EVERY, mode='valid')

        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward {SHOW_EVERY}ma")
        plt.xlabel("episode #")
        plt.show()
