#import datasets and libs
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Download latest version
import kagglehub
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

# AQIカテゴリ（6段階分類）
def categorize_aqi(aqi):
    if aqi <= 50:
        return 0  # Good
    elif aqi <= 100:
        return 1  # Moderate
    elif aqi <= 150:
        return 2  # Unhealthy for SG
    elif aqi <= 200:
        return 3  # Unhealthy
    elif aqi <= 300:
        return 4  # Very Unhealthy
    else:
        return 5  # Hazardous

df["aqi_category"] = df["aqi"].apply(categorize_aqi)

# 特徴量
X = df[["pm2_5", "pm10", "no2", "o3", "temperature", "humidity"]].values.astype(np.float32)
y = df["aqi_category"].values.astype(np.int64)

#split_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# torch tensor に変換
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# ニューラルネットワーク（分類用）
class ClassificationNN(torch.nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# モデル初期化
model = ClassificationNN(input_size=6, hidden1=32, hidden2=16, output_size=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

#train loop
for epoch in range(20):
    model.train()
    logits = model(X_train_tensor)
    loss = loss_fn(logits, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

#prediction
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    y_pred_tensor = torch.argmax(logits, dim=1)
    y_pred = y_pred_tensor.numpy()

# Accuracy and classification report
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 可視化（PM2.5 vs AQIクラス）
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], y_test, label="Actual", alpha=0.5)
plt.scatter(X_test[:, 0], y_pred, label="Predicted", alpha=0.5, color='red')
plt.xlabel("PM2.5")
plt.ylabel("AQI Category")
plt.title("Classification: PM2.5 and AQI Category (Neural Network)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
