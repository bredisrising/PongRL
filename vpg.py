from base_algo import Base
import torch

class VPG(Base):
    def __init__(self, batches, time_steps, lr=1e-3, df=.9999):
        super().__init__(batches, time_steps, lr, df)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def optimize(self):
        super().discount()

        loss = 0
        for b in range(self.batches_to_collect):
            probs = torch.stack(self.probs[b])
            returns = torch.tensor(self.returns[b], dtype=torch.float32)
            loss += (-probs * returns).sum()

        loss = loss / self.batches_to_collect

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        

        # loss = (-probs * returns)
        # print(loss)
        # loss.backward()
        # self.optimizer.step()
        # super().reset()