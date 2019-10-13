import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import datetime
import glob


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
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class DQN(object):

    def __init__(self, input_size, nb_action, gamma):
        self.capacity = 100000
        self.temperature = 100
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(self.capacity)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.last_state = torch.FloatTensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * self.temperature)
        action = probs.multinomial()
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state)\
            .gather(1, batch_action.unsqueeze(1))\
            .squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.FloatTensor(new_signal).unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.FloatTensor([self.last_reward]),
                          torch.LongTensor([int(self.last_action)]),
                          ))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        },
            os.path.join('saved_brain', datetime.datetime.strftime(datetime.datetime.now(), '%M%S%f.pth'))
        )

    def load(self):
        saved_brains = glob.glob(os.path.join('saved_brain', '*.pth'))
        saved_brains.sort(reverse=True)
        if len(saved_brains) == 0:
            print('No model to load')
        else:
            print('Loading model ' + saved_brains[0])
            checkpoint = torch.load(saved_brains[0])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

