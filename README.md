# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Load and preprocess the weather sensor data.

Extract relevant environmental and time features.

Split the data into training and testing sets.

Train Random Forest models for temperature, PM2.5, and energy.

Predict the outputs and calculate MAE, RMSE, and R².


## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: VIROSHAN S
RegisterNumber: 212224060304 
*/import pandas as pd
import numpy as np
from google.colab import files
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

uploaded = files.upload()
file = list(uploaded.keys())[0]

df = pd.read_csv(file)

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

df["hour"] = df["time"].dt.hour
df["dayofyear"] = df["time"].dt.dayofyear
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek

features = [
    "hum", "co2", "illumination", "pressure", "pm2_5",
    "pm10", "wind_direction_angle", "wind_speed",
    "wind_speed_level", "tsr", "hour", "dayofyear",
    "month", "dayofweek"
]

X = df[features].fillna(df[features].mean())

y_temp = df["tem"].fillna(df["tem"].mean())
y_pollution = df["pm2_5"].fillna(df["pm2_5"].mean())
y_energy = df["tsr"].fillna(df["tsr"].mean())

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train_temp = y_temp.iloc[:split]
y_test_temp = y_temp.iloc[split:]

y_train_pollution = y_pollution.iloc[:split]
y_test_pollution = y_pollution.iloc[split:]

y_train_energy = y_energy.iloc[:split]
y_test_energy = y_energy.iloc[split:]

temp_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

pollution_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

energy_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

temp_model.fit(X_train, y_train_temp)
pollution_model.fit(X_train, y_train_pollution)
energy_model.fit(X_train, y_train_energy)

temp_pred = temp_model.predict(X_test)
pollution_pred = pollution_model.predict(X_test)
energy_pred = energy_model.predict(X_test)

print("Random Forest Results")
print("---------------------")

print("\nTemperature Prediction")
print("MAE :", f"{mean_absolute_error(y_test_temp, temp_pred):.2f}")
print("RMSE:", f"{np.sqrt(mean_squared_error(y_test_temp, temp_pred)):.2f}")
print("R2  :", f"{r2_score(y_test_temp, temp_pred):.2f}")

print("\nPM2.5 Pollution Prediction")
print("MAE :", f"{mean_absolute_error(y_test_pollution, pollution_pred):.2f}")
print("RMSE:", f"{np.sqrt(mean_squared_error(y_test_pollution, pollution_pred)):.2f}")
print("R2  :", f"{r2_score(y_test_pollution, pollution_pred):.2f}")

print("\nEnergy Prediction")
print("MAE :", f"{mean_absolute_error(y_test_energy, energy_pred):.2f}")
print("RMSE:", f"{np.sqrt(mean_squared_error(y_test_energy, energy_pred)):.2f}")
print("R2  :", f"{r2_score(y_test_energy, energy_pred):.2f}")

```

## Output:

<img width="912" height="485" alt="Screenshot 2026-08-20 110337" src="https://github.com/user-attachments/assets/9df62248-47c6-45e7-8b02-b77d2d902297" />

## Result:
Thus,a python program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm has completed successfully.
