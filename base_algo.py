import torch
import torch.nn as nn
from torch.distributions import Categorical
class Base:
    def __init__(self, batches, time_steps, lr=1e-3, df=.99):

        self.policy = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax()
        )

        self.lr = lr
        self.df = df

        self.reset()

        self.batches_to_collect = batches
        self.time_steps = time_steps



    def reset(self):
        self.states = [[]]
        self.actions = [[]]
        self.probs = [[]]
        self.rewards = [[]]
    
    def reward(self, reward):
        self.rewards[-1].append((reward, len(self.states[-1])-1))

    def discount(self):
        for b in range(self.batches_to_collect):
            for r in self.rewards[b]:
                pass

    def add(self, state, action, prob):
        self.states[-1].append(state)
        self.actions[-1].append(action)
        self.probs[-1].append(prob)

        if len(self.states[-1]) >= self.time_steps:
            self.states.append([])
            self.actions.append([])
            self.probs.append([])
            
        if len(self.states) >= self.batches_to_collect:
            self.optimize()
            self.reset()

    def sample_action(self, state):
        dist = Categorical(self.policy(state))
        action = dist.sample()
        self.add(None, None, dist.log_prob(action))
        return action

    def optimize(self):
        pass
        


