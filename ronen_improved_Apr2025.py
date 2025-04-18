
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# === Vessel Cubic Curve Data (normally passed from vessel input section) ===
vessels = {
    "LNG Vessel A": {
        "coeffs": [15.2, -1.8, 0.9, 0.015],  # a, b, c, d
        "EF": 2.75
    },
    "LNG Vessel B": {
        "coeffs": [18.0, -1.5, 1.1, 0.012],
        "EF": 2.75
    }
}

# === UI Setup ===
st.set_page_config(page_title="Ronen Speed Optimization (Polynomial Fuel Curve)", layout="wide")
st.title("🚢 Ronen Model with Polynomial Fuel Curve & ETS Cost")

# Sidebar Inputs
with st.sidebar:
    st.header("Inputs")

    vessel_name = st.selectbox("Select Vessel", list(vessels.keys()))
    a, b, c, d = vessels[vessel_name]["coeffs"]
    EF = vessels[vessel_name]["EF"]

    L = st.number_input("Voyage Distance (nm)", value=4000)
    Dp = st.number_input("Port Days", value=2.0)
    Vmin = st.slider("Min Speed (knots)", 10.0, 17.0, 12.0)
    Vmax = st.slider("Max Speed (knots)", 17.0, 22.0, 19.0)
    C = st.slider("Daily Ops Cost ($)", 5000, 40000, 12000)
    Fc = st.slider("Fuel Cost ($/ton)", 300, 1200, 800)

    # ETS inputs
    CO2_price = st.slider("CO₂ Allowance Price ($/tCO₂)", 50, 200, 100)
    EU_share = st.slider("EU ETS Coverage (%)", 0, 100, 50)

    # Chartering inputs
    freight_rate = st.slider("Freight Rate ($/day)", 0, 200000, 100000, step=5000)
    assumed_speed = st.slider("Assumed Speed for Revenue", Vmin, Vmax, 15.0)
    Ca = st.number_input("Alternative Vessel Value ($/day)", value=70000)
    K = st.slider("Bonus/Penalty Rate ($/day)", 0, 50000, 25000)
    VR = st.slider("Reference Contract Speed (VR)", 10.0, 25.0, 18.0)

# === Calculations ===
V_range = np.linspace(Vmin, Vmax, 300)

def F(V):  # Fuel curve from cubic coefficients
    return a + b * V + c * V**2 + d * V**3

def ETS_cost_per_day(V):
    fuel = F(V)
    return fuel * EF * CO2_price * (EU_share / 100)

def Ds(V):  # Sea days
    return L / (24 * V)

def D(V):
    return Ds(V) + Dp

def R_total(V):
    return freight_rate * D(V)

def model1(V):
    fuel = F(V)
    total_cost = C * D(V) + fuel * (Fc + ETS_cost_per_day(V)) * Ds(V)
    return (R_total(V) - total_cost) / D(V)

def model2(V):
    fuel = F(V)
    return (Ca + fuel * Fc) * L / (24 * V)

def model3(V):
    fuel = F(V)
    R_adj = R_total(V) + (K * L / 24) * (1 / VR - 1 / V)
    total_cost = C * D(V) + fuel * (Fc + ETS_cost_per_day(V)) * Ds(V)
    return (R_adj - total_cost) / D(V)

# === Evaluate and Plot ===
Z1 = [model1(v) for v in V_range]
Z2 = [model2(v) for v in V_range]
Z3 = [model3(v) for v in V_range]

def find_optimum(Zlist):
    idx = np.argmax(Zlist) if Zlist != Z2 else np.argmin(Zlist)
    return V_range[idx], Zlist[idx]

V1_opt, Z1_opt = find_optimum(Z1)
V2_opt, Z2_opt = find_optimum(Z2)
V3_opt, Z3_opt = find_optimum(Z3)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📘 Model 1: Income Leg")
    st.markdown(f"**Optimum Speed:** {V1_opt:.2f} kn")
    st.markdown(f"**Daily Profit:** ${Z1_opt:,.0f}")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_range, y=Z1, mode="lines", name="Profit"))
    fig1.update_layout(title="Model 1: Profit vs Speed", xaxis_title="Speed (kn)", yaxis_title="Daily Profit ($)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📙 Model 2: Ballast")
    st.markdown(f"**Optimum Speed:** {V2_opt:.2f} kn")
    st.markdown(f"**Total Cost:** ${Z2_opt:,.0f}")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=V_range, y=Z2, mode="lines", name="Cost"))
    fig2.update_layout(title="Model 2: Cost vs Speed", xaxis_title="Speed (kn)", yaxis_title="Cost ($)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.subheader("📗 Model 3: Bonus/Penalty")
    st.markdown(f"**Optimum Speed:** {V3_opt:.2f} kn")
    st.markdown(f"**Daily Profit:** ${Z3_opt:,.0f}")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=V_range, y=Z3, mode="lines", name="Profit"))
    fig3.update_layout(title="Model 3: Profit vs Speed", xaxis_title="Speed (kn)", yaxis_title="Daily Profit ($)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
