import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
import random
from copy import deepcopy
class Base:
    def __init__(self, batches_to_collect, batch_size, time_steps, lr=1e-3, df=.99, load=False, p=None):
        
        if p is None:
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
        else:
            self.policy = p
        
        # self.old = deepcopy(self.policy)

        self.lr = lr
        self.df = df

        self.batch_size = batch_size
        self.batches_to_collect = batches_to_collect
        self.time_steps = time_steps

        self.update_counter = 0

        self.accumulated_reward = 0

        self.reset()

    
    def reset(self):
        self.actions = [[[]]]
        self.states = [[[]]]
        self.entropys = [[[]]]
        self.log_probs = [[[]]]
        self.probs = [[[]]]
        self.rewards = [[[]]]
        self.returns = [[[]]]
    
    def reward(self, reward):
        self.accumulated_reward += reward
        self.rewards[-1][-1].append((reward, len(self.states[-1][-1])-1))

    def discount(self):
        for b in range(self.batches_to_collect):
            for e in range(self.batch_size):
                for r in range(len(self.rewards[b][e])):
                    if r-1 < 0:
                        prev_time = -1
                    else:
                        prev_time = self.rewards[b][e][r-1][1]
                    reward, time = self.rewards[b][e][r]

                    for i, index in enumerate(range(prev_time+1, time+1)):
                        self.returns[b][e][index] = reward * self.df**(((time+1) - (prev_time+1)) - (i+1))


    def add(self, action, state, entropy, log_prob, prob):
        self.actions[-1][-1].append(action)
        self.states[-1][-1].append(state)
        self.entropys[-1][-1].append(entropy)
        self.log_probs[-1][-1].append(log_prob)
        self.probs[-1][-1].append(prob)
        self.returns[-1][-1].append(0)
        

        if len(self.states) >= self.batches_to_collect and len(self.states[-1]) >= self.batch_size and len(self.states[-1][-1]) >= self.time_steps:
            self.optimize()
            self.reset()

        if len(self.states[-1]) >= self.batch_size and len(self.states[-1][-1]) >= self.time_steps:
            self.actions.append([[]])
            self.states.append([[]])
            self.entropys.append([[]])
            self.log_probs.append([[]])
            self.probs.append([[]])
            self.rewards.append([[]])
            self.returns.append([[]])

        if len(self.states[-1][-1]) >= self.time_steps:
            self.actions[-1].append([])
            self.states[-1].append([])
            self.entropys[-1].append([])
            self.log_probs[-1].append([])
            self.probs[-1].append([])
            self.rewards[-1].append([])
            self.returns[-1].append([])

        #print(len(self.states), len(self.states[-1]), len(self.states[-1][-1]), '             ',  end="\r")

    def sample_action(self, state, p=False):
        output = self.policy(state)
        if p:
            print(output, end="\r")

        dist = Categorical(output)
        
        if random.random() < .1:
            action = dist.sample()
        else:
            action = torch.argmax(output)

        self.add(action, state, dist.entropy(), dist.log_prob(action), output[action])
        return action, output

        
    def optimize(self):
        pass

