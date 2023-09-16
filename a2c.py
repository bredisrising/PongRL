from base_algo import Base
import torch
import torch.nn as nn
#without value function
class A2C(Base):
    def __init__(self, name, batches, time_steps, lr=1e-3, df=.993, load=False):
        super().__init__(batches, time_steps, lr, df)
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
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr)
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


        ploss = 0
        entropy_loss = 0

        vloss = 0   

        for b in range(self.batches_to_collect):
            entropys = torch.stack(self.entropys[b])
            log_probs = torch.stack(self.log_probs[b])
            returns = torch.tensor([self.returns[b]], dtype=torch.float32).permute(1,0)
            advantages = torch.stack(self.advantages[b])
            vloss += self.value_loss_fn(self.values[b], returns.detach()).sum()
            ploss += (-log_probs * advantages.detach()).sum()
            entropy_loss -= entropys.sum()

        ploss = ploss / self.batches_to_collect
        vloss = vloss / self.batches_to_collect
        entropy_loss = entropy_loss / self.batches_to_collect

        self.policy_optimizer.zero_grad()
        loss = ploss + entropy_loss 
        loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        vloss.backward()
        self.value_optimizer.step()

        self.update_counter += 1