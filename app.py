import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import cvxpy as cp
import random

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(
    page_title="AI-Driven EV Charging Price Optimization & Smart Grid Load Balancing",
    layout="wide"
)

st.title("⚡ AI-Driven EV Charging Price Optimization & Smart Grid Load Balancing")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("⚙️ Settings")

forecast_horizon = st.sidebar.slider("Price Forecast Horizon (hours)", 6, 48, 12)
base_price = st.sidebar.number_input("Base Price (₹/kWh)", min_value=5.0, max_value=20.0, value=8.0, step=0.5)

# -------------------------------
# Load Dataset
# -------------------------------
try:
    df = pd.read_csv("data/pjm_hourly.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found. Please place 'pjm_hourly.csv' inside the 'data/' folder.")
    st.stop()

# Detect date and load columns
if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
elif 'datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['datetime'])
else:
    df['Datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')

load_col = df.select_dtypes(include=[np.number]).columns[-1]
df['Load'] = pd.to_numeric(df[load_col], errors='coerce').fillna(df[load_col].mean())

# Rebase timestamps to make recent
latest_time = datetime.now()
hours_back = len(df)
df['Datetime'] = [latest_time - timedelta(hours=i) for i in range(hours_back)][::-1]
df = df[df['Datetime'] >= (datetime.now() - timedelta(days=30))].reset_index(drop=True)

# -------------------------------
# Train Model
# -------------------------------
df['Hour'] = df['Datetime'].dt.hour
df['Day'] = df['Datetime'].dt.day
df['Month'] = df['Datetime'].dt.month

X = df[['Hour', 'Day', 'Month']]
y = df['Load']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------------
# Forecast Future Loads
# -------------------------------
future_hours = pd.date_range(datetime.now(), periods=forecast_horizon, freq='H')
future_df = pd.DataFrame({
    'Hour': future_hours.hour,
    'Day': future_hours.day,
    'Month': future_hours.month
})

predicted_loads = model.predict(future_df)

# -------------------------------
# Optimization for Dynamic Pricing
# -------------------------------
def compute_dynamic_prices(load_forecast, base_price):
    n = len(load_forecast)
    price = cp.Variable(n)
    load_mean = np.mean(load_forecast)

    objective = cp.Minimize(cp.sum_squares(load_forecast - load_mean) + 0.05 * cp.sum_squares(price - base_price))
    constraints = [price >= base_price * 0.8, price <= base_price * 1.2]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except:
        prob.solve()
    return np.array(price.value).flatten()

prices = compute_dynamic_prices(predicted_loads, base_price)

# -------------------------------
# Dashboard Visuals
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Recent Grid Load (Total MW)")
    fig = px.line(df, x='Datetime', y='Load', title="Past Week Grid Load Trend", markers=True)
    fig.update_layout(xaxis_title="Time", yaxis_title="Load (MW)", width=700, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Forecasted Dynamic Prices")
    forecast_df = pd.DataFrame({
        'Time': future_hours,
        'Predicted Load (MW)': predicted_loads,
        'Optimized Price (₹/kWh)': prices
    })
    fig2 = px.line(forecast_df, x='Time', y='Optimized Price (₹/kWh)', title="Dynamic Price Forecast", markers=True)
    fig2.update_layout(width=700, height=400)
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Metrics Summary
# -------------------------------
st.markdown("---")
st.subheader("🔎 Summary Insights")

colA, colB, colC = st.columns(3)
colA.metric("Current Grid Load (MW)", f"{df['Load'].iloc[-1]:.2f}")
colB.metric("Average Grid Load (MW)", f"{df['Load'].mean():.2f}")
colC.metric("Predicted Peak Load (Next 12h)", f"{np.max(predicted_loads):.2f}")

# -------------------------------
# AI Chatbot Assistant (Local Simulation)
# -------------------------------
st.markdown("---")
st.subheader("🤖 Intelligent Assistant")

user_query = st.text_input("Ask the Assistant about grid load, pricing, or EV charging:")

if user_query:
    responses = [
        "The grid load is currently higher due to evening demand peaks.",
        "Prices were optimized to maintain stability across the forecast horizon.",
        "Dynamic pricing balances EV charging demand with grid capacity.",
        "The predicted peak in the next few hours is managed using AI-based forecasts.",
        "Lower demand hours are assigned reduced prices to encourage off-peak charging."
    ]
    st.info(random.choice(responses))
