import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("TkAgg")  # GUIバックエンドの設定
import matplotlib.pyplot as plt

# --- データ読み込みと前処理 ---
df = pd.read_csv('../data/1553768847-housing.csv').dropna()

# ocean_proximityをOne-hotエンコーディング
df = pd.get_dummies(df, columns=["ocean_proximity"])

# 数値列選定と相関計算（median_house_valueと5%以上の相関のあるもの）
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
correlation_matrix = df[numeric_columns].corr()
correlation_with_target = correlation_matrix['median_house_value'].drop('median_house_value')
main_numeric = correlation_with_target[correlation_with_target.abs() > 0.05].index.tolist()

# 全てのOne-hot列も説明変数として追加
ocean_columns = [col for col in df.columns if col.startswith("ocean_proximity_")]
main_features = main_numeric + ocean_columns

print("5%以上の相関を持つ数値特徴量:", main_numeric)
print("One-hot特徴量:", ocean_columns)
print("最終的な説明変数:", main_features)

# 説明変数・目的変数
X = df[main_features].values.astype(np.float32)
y = df["median_house_value"].values.astype(np.float32).reshape(-1, 1)

# 標準化（数値だけでOK。one-hotは0/1なのでスケーリング不要）
scaler = StandardScaler()
X[:, :len(main_numeric)] = scaler.fit_transform(X[:, :len(main_numeric)])

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# NN定義
class ClassificationNN(torch.nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden1)
        # self.bn1 = torch.nn.BatchNorm1d(hidden1)
        # self.dp1 = torch.nn.Dropout(0.05)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        # self.bn2 = torch.nn.BatchNorm1d(hidden2)
        # self.dp2 = torch.nn.Dropout(0.05)
        self.fc3 = torch.nn.Linear(hidden2, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.bn1(x)
        # x = self.dp1(x)
        x = torch.relu(self.fc2(x))
        # x = self.bn2(x)
        # x = self.dp2(x)
        return self.fc3(x)

model = ClassificationNN(input_size=X.shape[1], hidden1=64, hidden2=32, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Early stopping設定
best_loss = float('inf')
patience = 500
patience_counter = 0

# 学習ループ
for epoch in range(50000):
    model.train()
    preds = model(X_train_tensor)
    loss = loss_fn(preds, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter > patience:
        print(f"Early stopping at epoch {epoch}")
        break

# 予測とclip
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()
    y_pred = np.clip(y_pred, 0, 500001)

# 評価
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy (1 - MAPE): {100 - mape:.2f}%")

# 可視化
plt.scatter(range(len(y_test)), y_test, label="True", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Pred", alpha=0.6, color='red')
plt.xlabel("Sample Index")
plt.ylabel("Median House Value")
plt.title("Prediction with Embedding + Numeric Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()