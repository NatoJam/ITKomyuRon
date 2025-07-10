#import datasets and libs
import os
import pandas as pd
import kagglehub
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("tfisthis/global-air-quality-and-respiratory-health-outcomes")

print("Path to dataset files:", path)
print("CSV files in directory:")
for file in os.listdir(path):
    if file.endswith(".csv"):
        print(file)

#read_csv
csv_file = os.path.join(path, "air_quality_health_dataset.csv")
df = pd.read_csv(csv_file)
print(df.head())

#notnull
print(df.isnull().sum())
df = df.dropna()

#PM2.5 with hospital_admissions
X = df[['pm2_5']].values.astype(np.float32)
y = df[['hospital_admissions']].values.astype(np.float32)
#split_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# torch tensor に変換
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# four-layer neural network
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

#model
model = SimpleNN(1, 16, 8, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

#train loop
for epoch in range(1000):
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

#prediction
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

#MSE and R²
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

#visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label="Actual", alpha=0.7)

#sort
sorted_idx = np.argsort(X_test.flatten())
plt.plot(X_test[sorted_idx], y_pred[sorted_idx], color="red", label="Predicted", linewidth=2)

plt.xlabel("PM2.5")
plt.ylabel("Hospital Admissions")
plt.title("Regression: PM2.5 and Hospital Admissions (Neural Network)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()