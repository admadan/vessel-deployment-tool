import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Calculation Functions ---
def calculate_freight_revenue(freight_rate_per_day, L, Dp, V):
    sea_days = L / (24 * V)
    total_days = sea_days + Dp
    revenue = freight_rate_per_day * total_days
    return revenue, total_days

def model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C):
    Z = []
    for V in V_range:
        Ds = L / (24 * V)
        D = Ds + Dp
        F = F0 * (V / V0) ** 3
        total_cost = C * D + F * Fc * Ds
        profit = R - total_cost
        Z.append(profit / D)
    return Z

def model2_cost_curve(V_range, Ca, V0, F0, Fc, L):
    Z = []
    for V in V_range:
        Ds = L / (24 * V)
        F = F0 * (V / V0) ** 3
        Z.append((Ca + F * Fc) * Ds)
    return Z

def model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR):
    Z = []
    for V in V_range:
        Ds = L / (24 * V)
        D = Ds + Dp
        bonus_penalty = (K * L / 24) * (1 / VR - 1 / V)
        adjusted_R = R + bonus_penalty
        F = F0 * (V / V0) ** 3
        total_cost = C * D + F * Fc * Ds
        profit = adjusted_R - total_cost
        Z.append(profit / D)
    return Z

def find_optimum(V_range, Z, mode='max'):
    Z = np.array(Z)
    idx = np.argmax(Z) if mode == 'max' else np.argmin(Z)
    return V_range[idx], Z[idx]

# --- Streamlit App ---
st.set_page_config(page_title="Ronen Optimal Speed Dashboard", layout="wide")
st.title("🚢 Ronen's Optimal Speed Dashboard")
st.markdown("Analyze optimal vessel speed based on different freight rates and voyage conditions.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("User Inputs")
    L = st.number_input("Voyage Distance (nm)", value=4000)
    V0 = st.number_input("Nominal Speed (knots)", value=19.0)
    F0 = st.number_input("Fuel Consumption @ V0 (tons/day)", value=120.0)
    Fc = st.number_input("Fuel Cost ($/ton)", value=800.0)
    Dp = st.number_input("Port Days", value=2)
    C = st.number_input("Daily Operating Cost ($)", value=12000.0)
    Ca = st.number_input("Alternative Daily Value of Ship ($)", value=70000.0)

    st.markdown("### Freight & Revenue Settings")
    freight_rate = st.number_input("Freight Rate ($/day)", value=100000)
    assumed_speed = st.slider("Assumed Speed for Revenue Calculation (knots)", min_value=10.0, max_value=V0, value=15.0)

    st.markdown("### Model 3 Settings")
    K = st.number_input("Penalty/Bonus per Day ($)", value=25000)
    VR = st.number_input("Reference Speed for Penalty/Bonus (knots)", value=18.0)

# --- Calculated Revenue ---
R, voyage_days = calculate_freight_revenue(freight_rate, L, Dp, assumed_speed)
st.markdown(f"""
### 📦 Freight Revenue Calculated
- Freight Rate: **${freight_rate:,.0f}/day**
- Assumed Speed: **{assumed_speed:.2f} knots**
- Voyage Duration: **{voyage_days:.2f} days**
- **Total Revenue: ${R:,.0f}**
""")

# --- Model Calculations ---
V_range = np.linspace(10, V0, 300)
Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)

V1_opt, Z1_opt = find_optimum(V_range, Z1, 'max')
V2_opt, Z2_opt = find_optimum(V_range, Z2, 'min')
V3_opt, Z3_opt = find_optimum(V_range, Z3, 'max')

# --- Central Graph ---
st.markdown("### 📊 Interactive Comparison of All Models")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(V_range, Z1, label="Model 1: Daily Profit", color='blue')
ax.plot(V_range, Z3, label="Model 3: Profit w/ Bonus/Penalty", linestyle='--', color='green')
ax.plot(V_range, Z2, label="Model 2: Total Cost", linestyle=':', color='orange')
ax.axhline(Ca, color='red', linestyle='-.', label=f"Alternative Value (${Ca:,.0f})")

# Mark optimal points
ax.plot(V1_opt, Z1_opt, 'o', color='blue', label=f"Model 1 Opt: {V1_opt:.2f} kn")
ax.plot(V2_opt, Z2_opt, 'o', color='orange', label=f"Model 2 Opt: {V2_opt:.2f} kn")
ax.plot(V3_opt, Z3_opt, 'o', color='green', label=f"Model 3 Opt: {V3_opt:.2f} kn")

ax.set_title("Daily Profit / Cost vs Speed")
ax.set_xlabel("Speed (knots)")
ax.set_ylabel("Daily Profit or Cost ($)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Recommendations ---
st.markdown("### ✅ Recommendations & Interpretation")
if Z1_opt < Ca and Z3_opt < Ca:
    st.error("Daily profit from both Model 1 and Model 3 is less than the ship's alternative value. Use Model 2 (cost minimization).")
elif Z1_opt > Z3_opt:
    st.success(f"Model 1 is more profitable at {V1_opt:.2f} knots (Z: ${Z1_opt:,.0f}).")
else:
    st.success(f"Model 3 is more profitable at {V3_opt:.2f} knots (Z: ${Z3_opt:,.0f}).")

st.markdown("Use the sidebar to change freight rate, vessel speed, and see how optimal speed and profits respond dynamically.")
