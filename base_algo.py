import torch
import torch.nn as nn


class Base:
    def __init__(self, batches, time_steps):

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

        self.lr = 1e-3
        self.df = .99

        self.states = [[]]
        self.actions = [[]]
        self.probs = [[]]
        self.rewards = [[0]]

        self.batches_to_collect = batches
        self.time_steps = time_steps

    def add(self, state, action, prob, reward):
        self.states[-1].append(state)
        self.actions[-1].append(action)
        self.probs[-1].append(prob)
        self.rewards[-1].append(reward)

        if len(self.states[-1]) >= self.time_steps:
            self.states.append([])
            self.actions.append([])
            self.probs.append([])
            self.rewards.append([0])
            
            

    def sample_action(self, state):
        return torch.multinomial(self.policy(state), 1).item()

    def optimize(self):
        if len(self.states) < self.batches_to_collect:
            return
        
