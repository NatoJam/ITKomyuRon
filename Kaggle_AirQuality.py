#import datasets and libs
import os
import pandas as pd
import kagglehub
from sklearn.linear_model import LinearRegression
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
X = df[['pm2_5']]
y = df['hospital_admissions']
#split_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#linear_regression
model = LinearRegression()
model.fit(X_train, y_train)
#prediction
y_pred = model.predict(X_test)

#MSE and R²
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

#visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label="Actual", alpha=0.7)

plt.plot(X_test, y_pred, color="red", label="Predicted", linewidth=2)

plt.xlabel("PM2.5")
plt.ylabel("Hospital Admissions")
plt.title("Regression: PM2.5 and Hospital Admissions")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()