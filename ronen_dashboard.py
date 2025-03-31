import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# === Model Calculation Functions ===
def model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C):
    Z_values = []
    for V in V_range:
        Ds = L / (24 * V)
        D = Ds + Dp
        F = F0 * (V / V0) ** 3
        total_cost = C * D + F * Fc * Ds
        profit = R - total_cost
        Z = profit / D
        Z_values.append(Z)
    return Z_values

def model2_cost_curve(V_range, Ca, V0, F0, Fc, L):
    Z_values = []
    for V in V_range:
        Ds = L / (24 * V)
        F = F0 * (V / V0) ** 3
        Z = (Ca + F * Fc) * Ds
        Z_values.append(Z)
    return Z_values

def model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR):
    Z_values = []
    for V in V_range:
        Ds = L / (24 * V)
        D = Ds + Dp
        bonus_penalty = (K * L / 24) * (1 / VR - 1 / V)
        adjusted_R = R + bonus_penalty
        F = F0 * (V / V0) ** 3
        total_cost = C * D + F * Fc * Ds
        profit = adjusted_R - total_cost
        Z = profit / D
        Z_values.append(Z)
    return Z_values

def find_optimum(V_range, Z_curve, mode='max'):
    Z_array = np.array(Z_curve)
    idx = np.argmax(Z_array) if mode == 'max' else np.argmin(Z_array)
    return V_range[idx], Z_array[idx]

# === Streamlit UI ===
st.title("🚢 Ronen's Optimal Speed Models")
st.markdown("Explore speed optimization under different freight market conditions using Models 1, 2, and 3.")

with st.sidebar:
    st.header("Input Parameters")
    L = st.number_input("Voyage Distance (nm)", value=4000)
    V0 = st.number_input("Nominal Speed (knots)", value=19.0)
    F0 = st.number_input("Fuel Consumption @ V0 (tons/day)", value=120.0)
    Fc = st.number_input("Fuel Cost ($/ton)", value=800.0)
    Dp = st.number_input("Port Days", value=2)
    C = st.number_input("Daily Operating Cost ($)", value=12000.0)
    Ca = st.number_input("Alternative Daily Value of Ship ($)", value=70000.0)

    st.subheader("Model 1 & 3 Inputs")
    R = st.slider("Freight Revenue ($)", min_value=500_000, max_value=2_000_000, value=1_200_000, step=50_000)

    st.subheader("Model 3 Specific")
    K = st.number_input("Penalty/Bonus per Day ($)", value=25000)
    VR = st.number_input("Reference Speed for Penalty/Bonus (knots)", value=18.0)

# Speed range
V_range = np.linspace(10, V0, 300)

# Compute curves
Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)

# Find optimal points
V1_opt, Z1_opt = find_optimum(V_range, Z1, 'max')
V2_opt, Z2_opt = find_optimum(V_range, Z2, 'min')
V3_opt, Z3_opt = find_optimum(V_range, Z3, 'max')

# Tabs for each model
tab1, tab2, tab3 = st.tabs(["📘 Model 1", "📙 Model 2", "📗 Model 3"])

with tab1:
    st.subheader("Model 1: Income-Generating Leg")
    fig, ax = plt.subplots()
    ax.plot(V_range, Z1, label="Daily Profit", color='blue')
    ax.axhline(Ca, color='red', linestyle='--', label=f"Alternative Value (${Ca:,.0f})")
    ax.plot(V1_opt, Z1_opt, 'ko', label=f"Opt. Speed: {V1_opt:.2f} kn\nZ: ${Z1_opt:,.0f}")
    ax.set_title("Model 1: Daily Profit vs Speed")
    ax.set_xlabel("Speed (knots)")
    ax.set_ylabel("Daily Profit ($)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    if Z1_opt < Ca:
        st.warning(f"Daily profit (${Z1_opt:,.2f}) is less than the alternative value (${Ca:,.2f}). Use Model 2 instead.")

with tab2:
    st.subheader("Model 2: Empty (Positioning) Leg")
    fig, ax = plt.subplots()
    ax.plot(V_range, Z2, label="Total Cost", color='orange')
    ax.plot(V2_opt, Z2_opt, 'ko', label=f"Opt. Speed: {V2_opt:.2f} kn\nZ: ${Z2_opt:,.0f}")
    ax.set_title("Model 2: Total Cost vs Speed")
    ax.set_xlabel("Speed (knots)")
