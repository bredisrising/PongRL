from base_algo import Base
import torch
import torch.nn as nn
#without value function
class A2C(Base):
    def __init__(self, name, batches, batch_size, time_steps, lr=1e-3, df=0.0, load=False, p=None):
        super().__init__(batches, batch_size, time_steps, lr, df, p)
        self.net = 'a2c'
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
        
        
        self.policy_path = "./models/"+name+"_a2c_policy.pt"
        self.value_path = "./models/"+name+"_a2c_value.pt"
        
        if load:
            self.policy.load_state_dict(torch.load(self.policy_path))  
            self.value.load_state_dict(torch.load(self.value_path))


        self.advantages = []
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=1.5e-3)
        self.value_loss_fn = nn.MSELoss()

    def save(self):
        torch.save(self.policy.state_dict(), self.policy_path)
        torch.save(self.value.state_dict(), self.value_path)

    def advantage(self):
        self.advantages = []
        self.values = []
        for b in range(self.batches_to_collect):
            self.values.append([])
            self.advantages.append([])
            for e in range(self.batch_size):
                self.values[b].append([])
                self.values[b][e] = self.value(torch.stack(self.states[b][e]))
                self.advantages[b].append([])
                # get next value - prev value for each index
                for i in range(len(self.values[b][e])):
                    if i == len(self.values[b][e]) - 1:
                        self.advantages[b][e].append(self.returns[b][e][i] - self.values[b][e][i])
                    else:
                        self.advantages[b][e].append(self.returns[b][e][i] + self.values[b][e][i+1]*.99 - self.values[b][e][i])

    def discount(self):
        self.advantages = []
        self.values = []
        for b in range(self.batches_to_collect):
            self.advantages.append([])
            self.values.append([])
            for e in range(self.batch_size):
                self.values[b].append([])
                self.values[b][e] = self.value(torch.stack(self.states[b][e]))
                self.advantages[b].append([])

                if len(self.rewards[b][e]) == 0 or self.rewards[b][e][-1][1] != self.time_steps-1:
                    self.rewards[b][e].append((0, self.time_steps-1))

                for r in range(len(self.rewards[b][e])):
                    if r-1 < 0:
                        prev_time = -1
                    else:
                        prev_time = self.rewards[b][e][r-1][1]
                    reward, time = self.rewards[b][e][r]

                    for i, index in enumerate(range(prev_time+1, time+1)):
                        self.returns[b][e][index] = reward * self.df**(((time+1) - (prev_time+1)) - (i+1))

                    for i, index  in enumerate(range(prev_time+1, time+1)):
                        if ((time+1) - (prev_time+1)) - (i+1) == 0:
                            #self.advantages[b][e][index] = self.returns[b][e][index] - self.values[b][e][index]
                            self.advantages[b][e].append(self.returns[b][e][index] - self.values[b][e][index])
                        else:
                            #self.advantages[b][e][index] = self.returns[b][e][index+1] + self.values[b][e][index+1]*.99 - self.values[b][e][index]
                            self.advantages[b][e].append(self.returns[b][e][index+1] + self.values[b][e][index+1]*.993 - self.values[b][e][index])

    def optimize(self):
        self.discount()

        ploss = 0
        entropy_loss = 0
        vloss = 0   

        for b in range(self.batches_to_collect):
            for e in range(self.batch_size):
                entropys = torch.stack(self.entropys[b][e])
                log_probs = torch.stack(self.log_probs[b][e])

                advantages = torch.stack(self.advantages[b][e])
                vloss += advantages.pow(2).sum()
                ploss += (-log_probs * advantages.detach()).sum()
                entropy_loss -= entropys.sum()

        # ploss = ploss / self.batches_to_collect
        # entropy_loss = entropy_loss / self.batches_to_collect
        vloss = vloss / self.batches_to_collect

        self.policy_optimizer.zero_grad()
        loss = ploss + entropy_loss * 0.01
        loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        vloss.backward()
        self.value_optimizer.step()

        self.update_counter += 1