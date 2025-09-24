import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# --- Page config ---
st.set_page_config(page_title="🌍 Climate What-If Explorer", layout="wide")
st.markdown("<h1 style='text-align:center;color:#2E8B57;'>🌿 Climate What-If Explorer</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Load data & model ---
@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_resource
def load_model(model_path, scaler_path):
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
    return model, scaler

# --- Paths ---
data_path = r"D:\game\climate_forecast_project\data\co2_monthly.csv"
model_path = r"D:\game\climate_forecast_project\models\lstm_model.pt"
scaler_path = r"D:\game\climate_forecast_project\models\scaler.pkl"

df = load_data(data_path)
model, scaler = load_model(model_path, scaler_path)

# --- Sidebar controls ---
st.sidebar.header("🌱 Change the World!")
planting_change = st.sidebar.slider("Plant more trees (%)", 0, 100, 20)
cars_change = st.sidebar.slider("Reduce cars (%)", 0, 100, 20)
forecast_months = st.sidebar.slider("Months to forecast 📆", 1, 24, 12)

# --- Forecast function ---
N = 12
last_seq_scaled = scaler.transform(df["value"].values[-N:].reshape(-1,1)).flatten()

def what_if_forecast(last_seq, months=12, planting_change=0, cars_change=0):
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

future_co2 = what_if_forecast(last_seq_scaled, months=forecast_months,
                               planting_change=planting_change, cars_change=cars_change)

# --- Main Visualization ---
st.subheader("📊 CO₂ Forecast Visualizer")

fig, ax = plt.subplots(figsize=(12,6))
months = np.arange(1, forecast_months+1)

# Dynamic multicolor bars
colors = sns.color_palette("rocket_r", forecast_months)
bars = ax.bar(months, future_co2, color=colors, edgecolor='black', alpha=0.9)

# Add circular markers with gradient sizes
for i, bar in enumerate(bars):
    ax.plot(bar.get_x() + bar.get_width()/2, bar.get_height(), 'o',
            color='yellow', markersize=8 + i*0.7, alpha=0.8)

ax.set_title("CO₂ Forecast for the Next Months", fontsize=18, weight='bold')
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("CO₂ (ppm)", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
st.pyplot(fig)

# --- Summary card ---
avg_change = future_co2[-1] - future_co2[0]
if avg_change < 0:
    st.markdown(f"<div style='padding:10px;background-color:#d4edda;color:#155724;border-radius:10px'>🌟 Awesome! CO₂ decreases by {abs(avg_change):.2f} ppm</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='padding:10px;background-color:#f8d7da;color:#721c24;border-radius:10px'>⚠️ CO₂ increases by {avg_change:.2f} ppm. Plant more trees! </div>", unsafe_allow_html=True)

# --- Month-wise Table ---
st.subheader("📅 Month-wise Forecast")
forecast_df = pd.DataFrame({"Month": months, "CO₂ (ppm)": future_co2})
st.dataframe(forecast_df.style.background_gradient(cmap='rocket_r').format("{:.2f}"))
