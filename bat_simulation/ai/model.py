import os
import random
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from constants import BATCH_SIZE, LEARNING_RATE, MODEL_FILE


class Network(nn.Module):

    def __init__(self, input_size: int, nb_action: int):
        super().__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # 2 layers fully connected network .
        # both layers are of size 30.
        hidden_layer_size = 64
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, nb_action)

    def forward(self, state):
        # Map state to q-value
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        # List of events
        self.memory = []
        self.position = 0

    def push(self, event: Tuple):
        """
        Push new event onto the memory stack.
        If the statck is full, delete the oldest event from the memory.

        event :  (self.last_state, new_state,
                  torch.LongTensor([int(self.last_action)]),
                  torch.Tensor([self.last_reward]))

                Note:
                  torch.LongTensor : dtype = int64
                  torch.Tensor : dtype = float32
        """
        # self.memory.append(event)
        # if len(self.memory) > self.capacity:
        #     del self.memory[0]
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*event)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List:
        """

        """
        # samples = zip(*random.sample(self.memory, batch_size))
        # return map(lambda x: Variable(torch.cat(x, 0)), samples)

        # List of transitions
        samples = random.sample(self.memory, batch_size)
        # Combine all elements with the same name into a tuple.
        # samples = zip(*samples)
        # # Each unpacked values are of type Variable
        # return map(lambda x: Variable(torch.cat(x, 0)), samples)
        return samples

    def __len__(self):
        return len(self.memory)


Transition = namedtuple('Transition',
                        ('last_state', 'new_state', 'last_action', 'last_reward'))


# Implementing Deep Q Learning

class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.nb_action = nb_action
        self.reward_window = []
        self.online_net = Network(input_size, nb_action)
        self.target_net = Network(input_size, nb_action)
        self.memory = ReplayMemory(BATCH_SIZE * 5)
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=LEARNING_RATE)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.epsilon = 0.05
        self.target_update = 100
        self.n_update = 0

    def _select_action(self, state: torch.Tensor):
        """
        state: 1D tensor containing [car.signal1, car.signal2,
                       car.signal3, orientation, -orientation]

        """

        sample = random.random()

        if sample < self.epsilon:
            # print("Play random")
            return random.randint(0, self.nb_action-1)
        # Predicted q_values for each of actions.
        # tensor of shape (1,41)
        q_vals = self.online_net.forward(state)
        # tensor of shape (1,41)
        probs = F.softmax(q_vals, dim=1)

        # action = probs.multinomial(1)
        action = probs.max(1)[1].view(1, 1)
        return action.data[0, 0]

    # def _learn(self, batch_state, batch_next_state, batch_reward, batch_action):
    #     outputs = self.online_net(batch_state).gather(
    #         1, batch_action.unsqueeze(1)).squeeze(1)

    #     # print("In learn")
    #     # print('batch_state')
    #     # print(batch_state.shape)
    #     # print(batch_state)
    #     # print('batch action ')
    #     # print(batch_action.shape)
    #     # print(batch_action)
    #     # print('batch_next_state')
    #     # print(batch_next_state.shape)
    #     # print(batch_next_state)
    #     # print('batch reward')
    #     # print(batch_reward.shape)
    #     # print(batch_reward)

    #     # print('Outputs')
    #     # print(outputs.shape)
    #     # print(outputs)
    #     next_outputs = self.online_net(batch_next_state).detach().max(1)[0]

    #     target = self.gamma*next_outputs + batch_reward

    #     td_loss = F.smooth_l1_loss(outputs, target)

    #     # td_loss.backward()
    #     td_loss.backward(retain_graph=True)
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()

    def update(self, reward, new_signal: List):
        """

        new_signal : List of features from the current state.
                    [car.signal1, car.signal2,
                       car.signal3, orientation, -orientation]
                       signal : int or float
                       orientation : float
        """
        # 1D tensor as the new_signel is always a list input.
        # Compress all features into an 1D tensor
        new_state = torch.Tensor(new_signal).unsqueeze(0)

        event = ((self.last_state, new_state, torch.LongTensor(
            [int(self.last_action)]), torch.Tensor([self.last_reward])))

        self.memory.push(event)

        action = self._select_action(new_state)

        if len(self.memory.memory) > BATCH_SIZE:
            # batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(
            #     BATCH_SIZE)

            # self._learn(batch_state, batch_next_state,
            #             batch_reward, batch_action)

            # List of transitions
            transitions = self.memory.sample(BATCH_SIZE)
            # Name tuple, where each name is correspond to tuple of size "batch_size" or "batch_size - 1"
            # Transition(last_state=(0, 1, 2, 3, 4), new_state=(1, 2, 3, 4, 5), last_action=(2, 3, 4, 5, 6), last_reward=(3, 4, 5, 6, 7))

            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.new_state)), dtype=torch.bool)

            non_final_next_states = torch.cat([s for s in batch.new_state
                                               if s is not None])

            # 2D Tensor size = [batch_size , input_size]
            state_batch = torch.cat(batch.last_state)
            # 1D Tensor size = [batch_size]
            action_batch = torch.cat(batch.last_action)
            # 1D Tensor size = [batch_size]
            reward_batch = torch.cat(batch.last_reward)
            # print("State_batch")
            # print(state_batch.shape)
            # print('Action')
            # print(action_batch.shape)

            # 2D Tensor size = [batch_size , ouput_size]
            state_action_values = self.online_net(
                state_batch)

            # Pickup the action values for acitons in the batch.
            state_action_values = state_action_values.gather(
                1, action_batch.unsqueeze(1)).squeeze(1)

            # print("Pred")
            # print(state_action_values.shape)
            # print(state_action_values)
            # state_action_values = self.online_net(
            #     state_batch).gather(1, action_batch)
            # print("Pred gather")
            # print(state_action_values.shape)
            # print(state_action_values)

            next_state_values = torch.zeros(BATCH_SIZE)

            # Here we pick up the max value of each row
            # 1D Tensor size = [batch_size]
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0].detach()

            # 1D Tensor size = [batch_size]
            expected_state_action_values = (
                next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values,
                                    expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.online_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        self.n_update += 1
        # Update the target network, copying all weights and biases in DQN
        if self.n_update % self.target_update == 0:
            # print("Update target")
            self.target_net.load_state_dict(self.online_net.state_dict())
        # print(action)
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self, model_file=MODEL_FILE):
        torch.save({'state_dict': self.online_net.state_dict(),
                    'target_net_dict': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, model_file)

    def load(self, model_file=MODEL_FILE):
        if os.path.isfile(model_file):
            print("=> loading checkpoint... ")
            checkpoint = torch.load(model_file)
            self.online_net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.target_net.load_state_dict(checkpoint['target_net_dict'])
            print("done !")
        else:
            print("no checkpoint found...")
