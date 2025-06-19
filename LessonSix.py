import torch
import numpy as np
class SingleNeuron(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l = torch.nn.Linear(input_size, output_size)
        self.out = torch.nn.ReLU()
    def forward(self, x):
        y = self.out(self.l(x))
        return y
def target_func(x):
    y = 0.5 * x[:,0] + 0.9 * x[:,1] + 0.3
    return y.reshape([-1,1])

model = SingleNeuron(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for epoch in range(1000):
    rnd = [[np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0)] for _ in range(100)]
    x = torch.tensor(rnd)
    
    y = model(x)
    y_hat = target_func(x)
    E = torch.nn.functional.mse_loss(y, y_hat, reduction="sum")

    optimizer.zero_grad()
    E.backward()
    optimizer.step()
print(model.l.weight, model.l.bias)
