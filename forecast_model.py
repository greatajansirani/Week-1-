# forecast_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("data/pjm_clean.csv", parse_dates=["Datetime"])
df = df.sort_values("Datetime")

# Feature engineering
df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek
df["Total_MW_lag1"] = df["Total_MW"].shift(1)
df["Total_MW_lag24"] = df["Total_MW"].shift(24)

df = df.dropna().reset_index(drop=True)

# target: next-hour Total_MW
df["target_next"] = df["Total_MW"].shift(-1)
df = df.dropna().reset_index(drop=True)

features = ["hour", "dayofweek", "Total_MW_lag1", "Total_MW_lag24"]
X = df[features]
y = df["target_next"]

# train/test split (time-series style — no shuffle)
split_idx = int(len(X)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"Test MAE: {mae:.2f} MW, R2: {r2:.3f}")

# save model
joblib.dump(model, "rf_model.joblib")
print("Saved rf_model.joblib")
