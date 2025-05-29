import numpy as np
import torch
class Regression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l=torch.nn.Linear(1, 1)
    def forward(self, x):
        y = self.l(x)
        return y
def target_func(x):
    return 0.5 * x + 0.3

model = Regression()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for epoch in range(1000):
    rnd = [[np.random.uniform(-1,1)] for _ in range(100)]
    x = torch.tensor(rnd)
    y = model(x)
    y_hat = target_func(x)
    E = torch.nn.functional.mse_loss(y, y_hat, reduction="sum")
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
print(model.l.weight, model.l.bias)