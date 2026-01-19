
import pandas as pd
import numpy as np
import glob # import csv files from the folder
import joblib # to save an load the trained model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split #split data into training and testing sets
from sklearn.metrics import mean_absolute_error #to evaluate the model, the error

csv_files = glob.glob("data/*.csv")

dfs = []

for file in csv_files:
    df = pd.read_csv(file)

    
# FEATURE ENGINEERING 

    df["dV_dt"] = df["Voltage_V"].diff() / df["Time_s"].diff()
    df["dV_dt"] = df["dV_dt"].fillna(0)

    end_time = df["Time_s"].iloc[-1]
    df["Remaining_Time_s"] = end_time - df["Time_s"]

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True) #combine all data

X = data[["Voltage_V", "Time_s", "dV_dt"]]
y = data["Remaining_Time_s"]


# TRAIN / TEST SPLIT data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN RANDOM FOREST REGRESSOR
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# EVALUATE MODEL
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} seconds")


joblib.dump(model, "battery_time_model.pkl") # save the model
print("Model saved as battery_time_model.pkl")