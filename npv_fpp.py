import streamlit as st
import numpy as np
import pandas as pd

# ---------------- Sidebar Inputs ----------------
st.set_page_config(page_title="Ship Speed NPV Optimizer", layout="wide")
st.sidebar.title("Input Parameters")

# General inputs
n = st.sidebar.number_input("Number of voyages (n)", 1, 10, value=3)
m = st.sidebar.number_input("Legs per voyage (m)", 1, 4, value=2)

# Route and fuel
distance = st.sidebar.number_input("Distance per leg (nm)", 1000, 20000, value=6000)
fuel_price = st.sidebar.number_input("Fuel price (USD/ton)", 100, 1000, value=600)
time_charter_cost = st.sidebar.number_input("Daily TCE cost (USD/day)", 5000, 50000, value=12000)
freight_rate = st.sidebar.number_input("Freight rate (USD/ton)", 1, 100, value=20)
cargo_quantity = st.sidebar.number_input("Cargo quantity (tons)", 10000, 100000, value=75000)
alpha = st.sidebar.number_input("Annual discount rate (%)", 0.0, 100.0, value=15.0) / 100 / 365
G0 = st.sidebar.number_input("Future Profit Potential G0 (USD)", 0, 10000000, value=700000)

# Speed range
v_min = st.sidebar.slider("Minimum speed (knots)", 10.0, 14.0, value=11.0)
v_max = st.sidebar.slider("Maximum speed (knots)", 14.0, 20.0, value=15.0)

# Fuel consumption polynomial coefficients for laden
st.sidebar.markdown("### Fuel Curve Coefficients - Laden")
a_l = st.sidebar.number_input("a (laden)", value=0.0)
b_l = st.sidebar.number_input("b (laden)", value=0.0)
c_l = st.sidebar.number_input("c (laden)", value=0.0)
d_l = st.sidebar.number_input("d (laden)", value=1.0)

# Fuel consumption polynomial coefficients for ballast
st.sidebar.markdown("### Fuel Curve Coefficients - Ballast")
a_b = st.sidebar.number_input("a (ballast)", value=0.0)
b_b = st.sidebar.number_input("b (ballast)", value=0.0)
c_b = st.sidebar.number_input("c (ballast)", value=0.0)
d_b = st.sidebar.number_input("d (ballast)", value=1.0)

# ---------------- Calculations ----------------
def fuel_consumption(v, is_laden):
    if is_laden:
        return a_l + b_l * v + c_l * v**2 + d_l * v**3
    else:
        return a_b + b_b * v + c_b * v**2 + d_b * v**3

def leg_time(v):
    return distance / (24 * v) + 2  # sailing time + 2 days port time

def discounted_profit(v, is_laden):
    T = leg_time(v)
    rev = freight_rate * cargo_quantity if is_laden else 0
    fuel_cost = fuel_consumption(v, is_laden) * T * fuel_price
    tce_cost = (1 - np.exp(-alpha * T)) * time_charter_cost / alpha
    return (rev * np.exp(-alpha * T)) - fuel_cost - tce_cost

speeds = np.arange(v_min, v_max + 0.1, 0.1)
data = []

for v in speeds:
    total_npv = 0
    total_time = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            is_laden = (j % 2 == 1)
            T = leg_time(v)
            profit = discounted_profit(v, is_laden)
            discount_factor = np.exp(-alpha * total_time)
            total_npv += profit * discount_factor
            total_time += T
    total_npv += G0 * np.exp(-alpha * total_time)
    data.append([round(v, 2), total_npv, total_time])

results = pd.DataFrame(data, columns=["Speed (knots)", "Total NPV (USD)", "Total Time (days)"])
opt_row = results.loc[results["Total NPV (USD)"].idxmax()]

# ---------------- UI Display ----------------
st.title("Ship Speed Optimization Dashboard")
st.subheader("NPV Maximization Based on Speed")

st.dataframe(results.style.format({"Total NPV (USD)": "${:,.0f}", "Total Time (days)": "{:.1f}"}))

st.markdown(f"### 🔧 Optimal Speed Recommendation")
st.markdown(f"- **Speed:** {opt_row['Speed (knots)']} knots\n- **Total NPV:** ${opt_row['Total NPV (USD)']:,.0f}\n- **Total Time:** {opt_row['Total Time (days)']:.1f} days")

st.line_chart(results.set_index("Speed (knots)")["Total NPV (USD)"])

st.caption("Model assumes simplified fuel and time functions. For detailed voyage modeling, integrate leg-specific data.")
