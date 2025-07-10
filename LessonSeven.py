import torch
import numpy as np
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, lower_hidden_size, upper_hidden_size, output_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, lower_hidden_size)
        self.l2 = torch.nn.Linear(lower_hidden_size, upper_hidden_size)
        self.l3 = torch.nn.Linear(upper_hidden_size, output_size)
    def forward(self, x):
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        o = self.l3(h2)
        return o

# class SimpleNN(torch.nn.Module):
#     def __init__(self, in_size, hid_size, out_size):
#         super().__init__()
#         self.l1 = torch.nn.Linear(in_size, hid_size)
#         self.l2 = torch.nn.Linear(hid_size, out_size)
#     def forward(self, x):
#         h = torch.relu(self.l1(x))
#         o = self.l2(h)
#         return o

def target_func(x):
    y = x[:,0] * x[:,0] + x[:,1] * 2.0
    return y.reshape([-1,1])

model = SimpleNN(2, 4, 3, 1)
#model = SimpleNN(2, 3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5000):
    rnd = [[np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)] for _ in range(100)]
    x = torch.tensor(rnd)
    y = model(x)
    y_hat = target_func(x)
    E = torch.nn.functional.mse_loss(y, y_hat, reduction="sum")
    optimizer.zero_grad()
    E.backward()
    optimizer.step()

rnd = [[np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)] for _ in range(10)]
x = torch.tensor(rnd)
y = model(x)
y_hat = target_func(x)
result = list(zip(y.data, y_hat.data))
for yy, yh in result:
    print(yy, yh)