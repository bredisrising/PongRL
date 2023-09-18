import torch
import torch.nn as nn

from base_algo import Base

class DQN(Base):
    def __init__(self, name, batches, batch_size, time_steps, lr=1e-3, df=.993, load=False, p=None):
        super().__init__(batches, batch_size, time_steps, lr, df, p=None)
        self.net  = 'dqn'
        neurons = 64

        self.qvalue = nn.Sequential(
            nn.Linear(5, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.Linear(neurons, 3),
        )

        self.qvalue_path = "./models/"+name+"_dqn_value.pt"

        self.qvalue_optimzier = torch.optim.Adam(self.qvalue.parameters(), lr=lr)
        self.qvalue_loss_fn = nn.MSELoss()

        self.epsilon = 1
        self.epsilon_decay = .95
        self.epsilon_min = .01

    def sample_action(self, state):
        output = self.qvalue(state)

        if torch.rand(1) < self.epsilon:
            action = torch.randint(0, 3, (1,))
        else:
            action = torch.argmax(output)

        self.add(action, state, None, None, torch.max(output))
        return action, output

    def save(self):
        torch.save(self.qvalue.state_dict(), self.qvalue_path)

    def optimize(self):
        super().discount()


        loss = 0
        for b in range(self.batches_to_collect):
            for e in range(self.batch_size):
                qvalues = torch.stack(self.probs[b][e]).unsqueeze(dim=1)
                returns = torch.tensor(self.returns[b][e], dtype=torch.float32).unsqueeze(dim=1)

                loss += self.qvalue_loss_fn(qvalues, returns).sum()

        loss = loss / self.batches_to_collect

        self.qvalue_optimzier.zero_grad()
        loss.backward()
        self.qvalue_optimzier.step() 

        self.update_counter += 1

        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        #print(self.epsilon, end="\r")