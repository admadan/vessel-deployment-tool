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
    return distance / (24 * v) + 2

def leg_profit(v, is_laden):
    T = leg_time(v)
    rev = freight_rate * cargo_quantity if is_laden else 0
    fuel_cost = fuel_consumption(v, is_laden) * T * fuel_price
    tce_cost = (1 - np.exp(-alpha * T)) * time_charter_cost / alpha
    return (rev * np.exp(-alpha * T)) - fuel_cost - tce_cost, T

# Dynamic programming algorithm
def dp_optimize(speeds):
    memo = {}  # (i, j, t): max NPV
    path = {}  # (i, j, t): best speed

    def dp(i, j, acc_time):
        if i == n and j == m:
            return G0 * np.exp(-alpha * acc_time), []

        key = (i, j, round(acc_time, 2))
        if key in memo:
            return memo[key]

        best_val = -np.inf
        best_speed = None
        best_seq = []

        for v in speeds:
            is_laden = (j % 2 == 1)
            profit, dt = leg_profit(v, is_laden)
            future_val, seq = dp(i + (j // m), (j % m) + 1, acc_time + dt)
            total_val = profit * np.exp(-alpha * acc_time) + future_val
            if total_val > best_val:
                best_val = total_val
                best_speed = v
                best_seq = [(i + 1, j + 1, round(acc_time + dt, 1), round(v, 2))] + seq

        memo[key] = (best_val, best_seq)
        return memo[key]

    return dp(0, 0, 0.0)

speeds = np.round(np.arange(v_min, v_max + 0.1, 0.1), 2)
npv_result, opt_path = dp_optimize(speeds)

# ---------------- UI Display ----------------
st.title("Ship Speed Optimization Dashboard")
st.subheader("NPV Maximization Using Dynamic Programming")

st.markdown(f"### 🔧 Optimal Strategy")
st.markdown(f"- **Total NPV:** ${npv_result:,.0f}")

if opt_path:
    st.markdown("#### Optimal Speed Plan per Leg")
    path_df = pd.DataFrame(opt_path, columns=["Voyage", "Leg", "Cumulative Time (days)", "Speed (knots)"])
    st.dataframe(path_df)

st.caption("Model implements NPV-based optimization using recursive dynamic programming across voyages and legs.")
