from base_algo import Base
import torch
import torch.nn as nn
#without value function
class VPG(Base):
    def __init__(self, name, batches, time_steps, lr=1e-3, df=.997, load=False):
        super().__init__(batches, time_steps, lr, df)
        
        self.name = name
        self.policy_path = "./models/"+name+"_vpg_policy.pt"
        if load:
            self.policy.load_state_dict(torch.load(self.policy_path))  

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
    

    def optimize(self):
        super().discount()

        ploss = 0

        for b in range(self.batches_to_collect):
            probs = torch.stack(self.probs[b])

            returns = torch.tensor(self.returns[b], dtype=torch.float32)

            ploss += (-probs * returns.detach()).sum()

        #ploss = ploss / self.batches_to_collect


        self.policy_optimizer.zero_grad()
        ploss.backward()
        self.policy_optimizer.step()

        self.update_counter += 1

    def save(self):
        torch.save(self.policy.state_dict(), self.policy_path)
