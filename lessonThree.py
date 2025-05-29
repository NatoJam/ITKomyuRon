import numpy as np
import torch

def eqn(x: float) -> float:
    return 5 * x ** 2 - 3 * x + 2

def deqn(x: float) -> float:
    return 10 * x - 3

alpha = 0.1
x = 0.1
for i in range(100):
    x = x - alpha * deqn(x)
print(f"eqn(x) = {eqn(x)} at x = {x}")

# PyTorchを使用して勾配を計算
alpha = 0.1
x = torch.tensor(0.1, requires_grad=True)

for i in range(100):
    E = 5 * x ** 2 - 3 * x + 2
    x.retain_grad()
    E.backward()
    x = x - alpha * x.grad
print(x)

# PyTorchの自動微分を使用して勾配を計算
alpha = 0.01
a = torch.tensor(0.1, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

for epoch in range(1000):
    rnd = np.random.normal(0, 1.0, 100)
    x = torch.tensor(rnd, requires_grad=True)
    y = a * x + b
    y_hat = 0.5 * x + 0.3
    E = torch.nn.functional.mse_loss(y, y_hat)

    a.retain_grad()
    b.retain_grad()
    E.backward()
    a = a - alpha * a.grad
    b = b - alpha * b.grad
    print(f"Epoch {epoch}: a = {a.item()}, b = {b.item()}, E = {E.item()}")
print(f"Final: a = {a.item()}, b = {b.item()}, E = {E.item()}")
print("a = %f, b = %f" % (a.item(), b.item()))
