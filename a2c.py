from base_algo import Base
import torch
import torch.nn as nn
#without value function
class A2C(Base):
    def __init__(self, batches, time_steps, lr=1e-3, df=.999):
        super().__init__(batches, time_steps, lr, df)

        neurons = 128
        self.value = nn.Sequential(
            nn.Linear(5, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 1),
        )
        
        
        self.advantages = []
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr)
        self.value_loss_fn = nn.MSELoss()

    def advantage(self):
        self.advantages = []
        for b in range(self.batches_to_collect):
            values = self.value(torch.stack(self.states[b]))
            self.advantages.append([])
            # get next value - prev value for each index
            for i in range(len(values)):
                if i == len(values) - 1:
                    self.advantages[b].append(values[i-1])
                else:
                    self.advantages[b].append(values[i+1] - values[i])

             


    def optimize(self):
        super().discount()
        self.advantage()

        ploss = 0
        vloss = 0
        for b in range(self.batches_to_collect):
            probs = torch.stack(self.probs[b])
            returns = torch.tensor([self.returns[b]], dtype=torch.float32).permute(1,0)
            advantages = torch.stack(self.advantages[b])
            vloss += self.value_loss_fn(self.value(torch.stack(self.states[b])), returns.detach()).sum()
            ploss += (-probs * advantages.detach()).sum()

        ploss = ploss / self.batches_to_collect
        vloss = vloss / self.batches_to_collect

        self.policy_optimizer.zero_grad()
        ploss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        vloss.backward()
        self.value_optimizer.step()