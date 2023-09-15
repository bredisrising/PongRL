import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
import random

class Base:
    def __init__(self, batches, time_steps, lr=1e-3, df=.99, load=False):

        neurons = 128
        self.policy = nn.Sequential(
            nn.Linear(5, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 3),
            nn.Softmax()
        )

        self.lr = lr
        self.df = df

        self.batches_to_collect = batches
        self.time_steps = time_steps

        self.update_counter = 0

        self.accumulated_reward = 0

        self.reset()

    
    def reset(self):
        self.states = [[]]
        self.actions = [[]]
        self.probs = [[]]
        self.rewards = [[]]
        self.returns = [[]]
    
    def reward(self, reward):
        self.accumulated_reward += reward
        self.rewards[-1].append((reward, len(self.states[-1])-1))

    def discount(self):
        for b in range(self.batches_to_collect):
            for r in range(len(self.rewards[b])):
                if r-1 < 0:
                    prev_time = -1
                else:
                    prev_time = self.rewards[b][r-1][1]
                reward, time = self.rewards[b][r]

                for i, index in enumerate(range(prev_time+1, time+1)):
                    self.returns[b][index] = reward * self.df**(((time+1) - (prev_time+1)) - (i+1))


    def add(self, state, action, prob):
        self.states[-1].append(state)
        self.actions[-1].append(action)
        self.probs[-1].append(prob)
        self.returns[-1].append(0)
        

        if len(self.states) >= self.batches_to_collect and len(self.states[-1]) >= self.time_steps:
            self.optimize()
            self.reset()


        if len(self.states[-1]) >= self.time_steps:
            self.states.append([])
            self.actions.append([])
            self.probs.append([])
            self.rewards.append([])
            self.returns.append([])

    def sample_action(self, state, p=False):
        output = self.policy(state)
        if p:
            print(output, end="\r")

        dist = Categorical(output)
        
        if random.random() < .1:
            action = dist.sample()
        else:
            action = torch.argmax(output)

        self.add(state, None, dist.log_prob(action))
        return action, output

        
    def optimize(self):
        pass

