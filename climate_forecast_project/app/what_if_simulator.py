import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler

data_path = r"D:\game\climate_forecast_project\data\co2_monthly.csv"
df = pd.read_csv(data_path, parse_dates=["date"])


model_path = r"D:\game\climate_forecast_project\models\lstm_model.pt"
scaler_path = r"D:\game\climate_forecast_project\models\scaler.pkl"

scaler = joblib.load(scaler_path)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel()
model.load_state_dict(torch.load(model_path))
model.eval()

def what_if_forecast(last_seq, months=12, planting_change=0, cars_change=0):
    """
    last_seq: numpy array of last N months CO2 values
    planting_change: % change in planting rate (e.g., 10 → +10%)
    cars_change: % change in car usage (e.g., -20 → reduce 20%)
    """
    seq = last_seq.copy()
    predictions = []

    for _ in range(months):
    
        adjustment = -0.02 * planting_change + 0.01 * cars_change 
        input_seq = torch.from_numpy((seq + adjustment).reshape(1, seq.shape[0], 1)).float()
        with torch.no_grad():
            pred_scaled = model(input_seq).numpy()
        pred_value = scaler.inverse_transform(pred_scaled.reshape(-1,1))[0,0]
        predictions.append(pred_value)
     
        seq = np.append(seq[1:], scaler.transform([[pred_value]])[0,0])
    
    return predictions

N = 12
last_seq_scaled = scaler.transform(df["value"].values[-N:].reshape(-1,1)).flatten()

future_co2 = what_if_forecast(last_seq_scaled, months=12, planting_change=20, cars_change=-10)

print("Next 12 months CO2 forecast with scenario:")
for i, val in enumerate(future_co2, 1):
    print(f"Month {i}: {val:.2f} ppm")
