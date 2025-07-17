# url: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot/data
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

# 前処理
df = pd.read_csv("../data/melb_data.csv")
df = df.dropna() # 欠損値削除

# 数値列を取る
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('Price')  # 目的変数を除外

# 相関係数を計算
correlation = df[numeric_columns + ['Price']].corr()
correlation_with_price = correlation['Price'].drop('Price')
for col in numeric_columns:
    print(f"{col}とPriceとの相関係数: {correlation_with_price[col]:.4f}")
selected_features = correlation_with_price[correlation_with_price.abs() >= 0.05].index.tolist()

print("相関関係が高い(5%以上)特徴量:", selected_features)

# 説明変数と目的変数
X = df[selected_features].values.astype(np.float32)
y = df['Price'].values.astype(np.float32).reshape(-1, 1)

# 標準化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# Tensor変換
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# NNの定義
class RegressionNN(torch.nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden1)
        self.dp1 = torch.nn.Dropout(0.05)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.dp2 = torch.nn.Dropout(0.05)
        self.fc3 = torch.nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dp1(x)
        x = torch.relu(self.fc2(x))
        x = self.dp2(x)
        return self.fc3(x)

# モデル構築
model = RegressionNN(input_size=X.shape[1], hidden1=64, hidden2=32, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# EarlyStop設定
best_loss = float('inf')
patience = 1000
patience_counter = 0

# 学習ループ
for epoch in range(15000):
    model.train()
    preds = model(X_train_tensor)
    loss = loss_fn(preds, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter > patience:
        print(f"Early stopping at epoch {epoch}")
        break

# 予測と評価
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

# スケーリングを戻す
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# 評価指標
mse = mean_squared_error(y_test_original, y_pred)
mape = mean_absolute_percentage_error(y_test_original, y_pred) * 100

print(f"MSE: {mse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {100 - mape:.2f}%")

# 可視化
plt.scatter(range(len(y_test_original)), y_test_original, label="True Price", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Price", alpha=0.6, color='red')
plt.xlabel("Selected_Features")
plt.ylabel("Price")
plt.title("True vs Predicted Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 任意入力による予測 ---
print("\n--- 任意の入力による予測 ---")

# 選ばれた特徴量名の確認
print("使用された特徴量:", selected_features)

# 検証
input_values = []
for feature in selected_features:
    val = float(input(f"Input {feature} value: "))
    input_values.append(val)

# 入力
input_array = np.array(input_values).reshape(1, -1)
input_scaled = scaler_X.transform(input_array)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# 予測
model.eval()
with torch.no_grad():
    pred_scaled = model(input_tensor).numpy()
    pred_price = scaler_y.inverse_transform(pred_scaled)

# 結果表示
print(f"\ny_pred: {pred_price[0][0]:,.2f}（AUD）")