import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
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
def load_lstm_model(model_path, scaler_path):
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
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, scaler

@st.cache_resource
def load_prophet_model(model_path):
    from prophet import Prophet
    return joblib.load(model_path)

# --- Paths (relative for GitHub/Streamlit Cloud) ---
data_path = "data/co2_monthly.csv"
lstm_model_path = "models/lstm_model.pt"
scaler_path = "models/scaler.pkl"
prophet_model_path = "models/prophet_model.pkl"

# --- Load data ---
df = load_data(data_path)

# --- Sidebar controls ---
st.sidebar.header("🌱 Change the World!")
model_choice = st.sidebar.radio("Choose model:", ["LSTM", "Prophet"])
planting_change = st.sidebar.slider("Plant more trees (%)", 0, 100, 20)
cars_change = st.sidebar.slider("Reduce cars (%)", 0, 100, 20)
forecast_months = st.sidebar.slider("Months to forecast 📆", 1, 24, 12)

# --- Forecast functions ---
def lstm_forecast(df, months, planting_change, cars_change):
    model, scaler = load_lstm_model(lstm_model_path, scaler_path)
    N = 12
    last_seq_scaled = scaler.transform(df["value"].values[-N:].reshape(-1, 1)).flatten()

    seq = last_seq_scaled.copy()
    predictions = []

    for _ in range(months):
        adjustment = -0.02 * planting_change + 0.01 * cars_change
        input_seq = torch.from_numpy((seq + adjustment).reshape(1, seq.shape[0], 1)).float()
        with torch.no_grad():
            pred_scaled = model(input_seq).numpy()
        pred_value = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        predictions.append(pred_value)
        seq = np.append(seq[1:], scaler.transform([[pred_value]])[0, 0])

    return predictions

def prophet_forecast(df, months, planting_change, cars_change):
    model = load_prophet_model(prophet_model_path)
    future = model.make_future_dataframe(periods=months, freq="M")
    forecast = model.predict(future)

    # Apply simple adjustments
    forecast["yhat"] += (-0.02 * planting_change + 0.01 * cars_change) * np.arange(len(forecast))

    return forecast.tail(months)["yhat"].values

# --- Run forecast ---
if model_choice == "LSTM":
    future_co2 = lstm_forecast(df, forecast_months, planting_change, cars_change)
else:
    future_co2 = prophet_forecast(df, forecast_months, planting_change, cars_change)

# --- Main Visualization ---
st.subheader("📊 CO₂ Forecast Visualizer")

fig, ax = plt.subplots(figsize=(12, 6))
months = np.arange(1, forecast_months + 1)

colors = sns.color_palette("rocket_r", forecast_months)
bars = ax.bar(months, future_co2, color=colors, edgecolor='black', alpha=0.9)

for i, bar in enumerate(bars):
    ax.plot(bar.get_x() + bar.get_width()/2, bar.get_height(), 'o',
            color='yellow', markersize=8 + i*0.7, alpha=0.8)

ax.set_title(f"CO₂ Forecast for the Next {forecast_months} Months ({model_choice})", fontsize=18, weight='bold')
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("CO₂ (ppm)", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
st.pyplot(fig)

# --- Summary card ---
avg_change = future_co2[-1] - future_co2[0]
if avg_change < 0:
    st.markdown(f"<div style='padding:10px;background-color:#d4edda;color:#155724;border-radius:10px'>🌟 Awesome! CO₂ decreases by {abs(avg_change):.2f} ppm</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='padding:10px;background-color:#f8d7da;color:#721c24;border-radius:10px'>⚠️ CO₂ increases by {avg_change:.2f} ppm. Plant more trees!</div>", unsafe_allow_html=True)

# --- Month-wise Table ---
st.subheader("📅 Month-wise Forecast")
forecast_df = pd.DataFrame({"Month": months, "CO₂ (ppm)": future_co2})
st.dataframe(forecast_df.style.background_gradient(cmap='rocket_r').format("{:.2f}"))
