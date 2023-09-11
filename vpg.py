import torch
import torch.nn as nn


class VPG:
    def __init__(self):

        self.policy = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        self.lr = 1e-3
        self.df = .99



    def sample_action(self, state):
        pass

    def optimize(self):
        pass