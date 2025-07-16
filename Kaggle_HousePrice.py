import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error
import matplotlib
matplotlib.use("TkAgg")  # GUIバックエンドの設定
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

model = 0  # モデルの初期化

# データ読み込み
df = pd.read_csv('../data/1553768847-housing.csv')
df = df.dropna()  # 欠損値削除
# 数値列のみ選択
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# 相関係数を計算
correlation_matrix = df[numeric_columns].corr()
correlation_with_target = correlation_matrix['median_house_value'].drop('median_house_value')

# 結果表示
print("median_house_valueとの相関係数:\n", correlation_with_target.sort_values(ascending=False))

# 相関係数が10%以上の特徴量を抽出
main_features = correlation_with_target[correlation_with_target.abs() > 0.05].index.tolist()
print("\n5%以上の相関を持つ特徴量:", main_features)

# 説明変数と目的変数
X = df[main_features].values.astype(np.float32)
y = df["median_house_value"].values.astype(np.float32).reshape(-1, 1)

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tensor変換
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# NN定義（指定の形式）
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

# モデル構築
model = ClassificationNN(input_size=X.shape[1], hidden1=64, hidden2=32, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

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

# 予測と評価
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# print(f"MSE: {mse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"accuracy: {(np.mean(1-(np.abs((y_test - y_pred) / y_test))) * 100):.2f}%")

# 可視化
plt.scatter(range(len(y_test)), y_test, label="True Value", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Value", alpha=0.6, color='red')
plt.xlabel("Sample Index")
plt.ylabel("Median House Value")
plt.title("True vs Predicted Median House Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()