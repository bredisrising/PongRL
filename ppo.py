from base_algo import Base

import torch
import torch.nn as nn

from copy import deepcopy

class PPO(Base):
    def __init__(self, name, batches, time_steps, lr=1e-3, df=.993, load=False):
        super().__init__(batches, time_steps, lr, df)
        self.net = 'ppo'
        neurons = 64
        self.value = nn.Sequential(
            nn.Linear(5, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.Linear(neurons, 1),
        )

        self.policy_path = "./models/"+name+"_ppo_policy.pt"
        self.value_path = "./models/"+name+"_ppo_value.pt"

        if load:
            self.policy.load_state_dict(torch.load(self.policy_path))  
            self.value.load_state_dict(torch.load(self.value_path))

        self.epsilon = 0.2

        self.advantages = []
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr/3)
        
        self.value_optimizer = torch.optim.Adam(self.value.parameters())
        self.value_loss_fn = nn.MSELoss()

    
    def save(self):
        torch.save(self.policy.state_dict(), self.policy_path)
        torch.save(self.value.state_dict(), self.value_path)

    def advantage(self):
        self.advantages = []
        self.values = []
        for b in range(self.batches_to_collect):
            self.values.append([])
            self.values[b] = self.value(torch.stack(self.states[b]))
            self.advantages.append([])
            # get next value - prev value for each index
            for i in range(len(self.values[b])):
                if i == len(self.values[b]) - 1:
                    self.advantages[b].append(self.returns[b][i] - self.values[b][i])
                else:
                    self.advantages[b].append(self.returns[b][i] + self.values[b][i+1]*.99 - self.values[b][i])

    def optimize(self):
        super().discount()
        self.advantage()

        # if self.update_counter == 0:
        #     self.update_counter += 1
        #     return
        
        # ploss = 0
        # vloss = 0   
        # entropy_loss = 0

        old_policy = deepcopy(self.policy)


        

        # ploss = 0
        # vloss = 0
        for b in range(self.batches_to_collect):
            #print('fuck'+str(b))
            old_probs = old_policy(torch.stack(self.states[b]).detach()).gather(1, torch.tensor(self.actions[b]).unsqueeze(1)).squeeze(1)
            cur_probs = self.policy(torch.stack(self.states[b]).detach()).gather(1, torch.tensor(self.actions[b]).unsqueeze(1)).squeeze(1)
            values = self.value(torch.stack(self.states[b]).detach())

            #entropys = torch.stack(self.entropys[b])
            returns = torch.tensor([self.returns[b]], dtype=torch.float32).permute(1,0)
            advantages = torch.stack(self.advantages[b])
            vloss = self.value_loss_fn(values, returns.detach()).mean()
            
            ratio = cur_probs / old_probs.detach()
            epic = ratio * advantages.detach()
            clip = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages.detach()
            ploss = -torch.min(epic, clip).mean()
            #entropy_loss = entropys.mean()
                
            # ploss = ploss / (self.batches_to_collect * self.time_steps)
            # vloss = vloss / (self.batches_to_collect * self.time_steps)

            self.policy_optimizer.zero_grad()
            (ploss).backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            vloss.backward()
            self.value_optimizer.step()

        self.update_counter += 1