
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar

st.set_page_config(page_title="Ronen Polynomial Dashboard – Models 1, 2, 3", layout="wide")
st.title("🚢 Ronen Models with Polynomial Fuel Curve")

# Sidebar Inputs
with st.sidebar:
    st.header("Fuel Curve Coefficients (F(V) = a + bV + cV² + dV³)")
    a = st.number_input("Coefficient a", value=15.0)
    b = st.number_input("Coefficient b", value=-1.5)
    c = st.number_input("Coefficient c", value=0.9)
    d = st.number_input("Coefficient d", value=0.015)

    st.header("Operational Inputs")
    Fc = st.number_input("Fuel Cost ($/ton)", value=800)
    C = st.number_input("Daily Ops Cost ($)", value=12000)
    L = st.number_input("Voyage Distance (nm)", value=4000)
    Dp = st.number_input("Port Days", value=2.0)

    st.header("Charter & Contract")
    freight_rate = st.number_input("Freight Rate ($/day)", value=100000)
    Ca = st.number_input("Alternative Vessel Value ($/day)", value=70000)
    K = st.number_input("Bonus/Penalty ($/day)", value=25000)
    VR = st.number_input("Reference Speed (VR)", value=18.0)
    assumed_speed = st.slider("Assumed Speed (kn)", 10.0, 20.0, 15.0)

# Speed range
V_range = np.linspace(10, 20, 300)

# Fuel function and helper functions
def F(V): return a + b*V + c*V**2 + d*V**3
def Ds(V): return L / (24 * V)
def D(V): return Ds(V) + Dp
def R(V): return freight_rate * D(V)

# Model 1: Income Leg
def Z1(V): return (R(V) - C * D(V) - F(V) * Fc * Ds(V)) / D(V)

# Model 2: Ballast Leg
def Z2(V): return (Ca + F(V) * Fc) * L / (24 * V)

# Model 3: Bonus/Penalty
def R_adj(V): return R(V) + (K * L / 24) * (1 / VR - 1 / V)
def Z3(V): return (R_adj(V) - C * D(V) - F(V) * Fc * Ds(V)) / D(V)

# Evaluate models
Z1_vals = [Z1(v) for v in V_range]
Z2_vals = [Z2(v) for v in V_range]
Z3_vals = [Z3(v) for v in V_range]

# Optimization
opt1 = minimize_scalar(lambda v: -Z1(v), bounds=(10, 20), method='bounded')
opt2 = minimize_scalar(Z2, bounds=(10, 20), method='bounded')
opt3 = minimize_scalar(lambda v: -Z3(v), bounds=(10, 20), method='bounded')

V1_opt, Z1_opt = opt1.x, Z1(opt1.x)
V2_opt, Z2_opt = opt2.x, Z2(opt2.x)
V3_opt, Z3_opt = opt3.x, Z3(opt3.x)

Z1_assumed = Z1(assumed_speed)
Z2_assumed = Z2(assumed_speed)
Z3_assumed = Z3(assumed_speed)

# Results Section
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("📘 Model 1: Fixed Revenue")
    st.markdown(f"- **Optimum Speed:** {V1_opt:.2f} kn")
    st.markdown(f"- **Daily Profit (Z):** ${Z1_opt:,.0f}")
    st.markdown(f"- **% Improvement:** {(Z1_opt - Z1_assumed) / Z1_assumed * 100:.2f}%")
    st.markdown(f"- **Savings:** ${(Z1_opt - Z1_assumed) * D(V1_opt):,.0f}")
with col2:
    st.subheader("📙 Model 2: Ballast Leg")
    st.markdown(f"- **Optimum Speed:** {V2_opt:.2f} kn")
    st.markdown(f"- **Total Cost (Z):** ${Z2_opt:,.0f}")
    st.markdown(f"- **% Cost Reduction:** {(Z2_assumed - Z2_opt) / Z2_assumed * 100:.2f}%")
    st.markdown(f"- **Savings:** ${(Z2_assumed - Z2_opt):,.0f}")
with col3:
    st.subheader("📗 Model 3: Bonus/Penalty")
    st.markdown(f"- **Optimum Speed:** {V3_opt:.2f} kn")
    st.markdown(f"- **Daily Profit (Z):** ${Z3_opt:,.0f}")
    st.markdown(f"- **% Improvement:** {(Z3_opt - Z3_assumed) / Z3_assumed * 100:.2f}%")
    st.markdown(f"- **Savings:** ${(Z3_opt - Z3_assumed) * D(V3_opt):,.0f}")

# Separate charts
st.subheader("📈 Model 1: Daily Profit")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=V_range, y=Z1_vals, mode="lines", name="Model 1", line=dict(color="blue")))
fig1.add_vline(x=V1_opt, line_dash="dash", line_color="blue", annotation_text=f"Zopt: ${Z1_opt:,.0f}")
fig1.add_vline(x=assumed_speed, line_dash="dot", line_color="gray", annotation_text=f"Zassumed: ${Z1_assumed:,.0f}")
fig1.update_layout(xaxis_title="Speed (kn)", yaxis_title="Z ($/day)", template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📈 Model 2: Total Cost")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=V_range, y=Z2_vals, mode="lines", name="Model 2", line=dict(color="orange")))
fig2.add_vline(x=V2_opt, line_dash="dash", line_color="orange", annotation_text=f"Zopt: ${Z2_opt:,.0f}")
fig2.add_vline(x=assumed_speed, line_dash="dot", line_color="gray", annotation_text=f"Zassumed: ${Z2_assumed:,.0f}")
fig2.update_layout(xaxis_title="Speed (kn)", yaxis_title="Z ($)", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📈 Model 3: Profit with Bonus/Penalty")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=V_range, y=Z3_vals, mode="lines", name="Model 3", line=dict(color="green")))
fig3.add_vline(x=V3_opt, line_dash="dash", line_color="green", annotation_text=f"Zopt: ${Z3_opt:,.0f}")
fig3.add_vline(x=assumed_speed, line_dash="dot", line_color="gray", annotation_text=f"Zassumed: ${Z3_assumed:,.0f}")
fig3.update_layout(xaxis_title="Speed (kn)", yaxis_title="Z ($/day)", template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)
