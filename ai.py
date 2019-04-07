import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

capacity = 100000
temperature = 7


class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action

        self.fc1 = nn.Linear(in_features=input_size, out_features=30)
        self.fc2 = nn.Linear(in_features=30, out_features=nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = F.softmax(self.fc2(x))

        return q_values


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0), samples))


class DQN(object):

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        self.last_state = torch.FloatTensor().unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * temperature)
        action = probs.multinomial()
        return action.data[0,0]
