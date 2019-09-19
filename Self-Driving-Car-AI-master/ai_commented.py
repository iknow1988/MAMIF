import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from typing import List, Tuple


BATCH_SIZE = 100
BRIAN_OUTPUT_FILE = 'tmp2_brain.pth'


class Network(nn.Module):

    def __init__(self, input_size: int, nb_action: int):
        super().__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # 2 layers fully connected network .
        # both layers are of size 30.
        hidden_layer_size = input_size * 2
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
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """

        """
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q Learning

class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def _select_action(self, state: torch.Tensor):
        """
        state: 1D tensor containing [car.signal1, car.signal2,
                       car.signal3, orientation, -orientation]

        """
        # Predicted q_values for each of actions.
        # tensor of shape (1,41)
        q_vals = self.model.forward(state)
        # tensor of shape (1,41)
        probs = F.softmax(q_vals, dim=1)

        # action = probs.multinomial(1)
        action = probs.max(1)[1].view(1, 1)
        return action.data[0, 0]

    def _learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(
            1, batch_action.unsqueeze(1)).squeeze(1)

        # print("In learn")
        # print('batch_state')
        # print(batch_state.shape)
        # print(batch_state)
        # print('batch action ')
        # print(batch_action.shape)
        # print(batch_action)
        # print('batch_next_state')
        # print(batch_next_state.shape)
        # print(batch_next_state)
        # print('batch reward')
        # print(batch_reward.shape)
        # print(batch_reward)

        # print('Outputs')
        # print(outputs.shape)
        # print(outputs)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]

        target = self.gamma*next_outputs + batch_reward

        td_loss = F.smooth_l1_loss(outputs, target)

        # td_loss.backward()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

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

        self.memory.push((self.last_state, new_state, torch.LongTensor(
            [int(self.last_action)]), torch.Tensor([self.last_reward])))

        action = self._select_action(new_state)

        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(
                BATCH_SIZE)

            self._learn(batch_state, batch_next_state,
                        batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, BRIAN_OUTPUT_FILE)

    def load(self):
        if os.path.isfile(BRIAN_OUTPUT_FILE):
            print("=> loading checkpoint... ")
            checkpoint = torch.load(BRIAN_OUTPUT_FILE)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
