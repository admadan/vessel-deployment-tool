
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar

st.set_page_config(page_title="Ronen Polynomial Models – Final Version", layout="wide")
st.title("🚢 Ronen Optimal Speed Dashboard – Final Version")

# Sidebar Inputs
with st.sidebar:
    st.header("Polynomial Fuel Curve (F(V) = a + bV + cV² + dV³)")
    a = st.number_input("a", value=15.0)
    b = st.number_input("b", value=-1.5)
    c = st.number_input("c", value=0.9)
    d = st.number_input("d", value=0.015)

    st.header("Operational Parameters")
    Fc = st.number_input("Fuel Cost ($/ton)", value=800)
    C = st.number_input("Daily Ops Cost ($)", value=12000)
    L = st.number_input("Voyage Distance (nm)", value=4000)
    Dp = st.number_input("Port Days", value=2.0)

    st.header("Charter/Contract Info")
    freight_rate = st.number_input("Freight Rate ($/day)", value=100000)
    Ca = st.number_input("Alternative Value ($/day)", value=70000)
    K = st.number_input("Bonus/Penalty ($)", value=25000)
    VR = st.number_input("Reference Speed (VR)", value=18.0)
    assumed_speed = st.slider("Assumed Speed", 10.0, 20.0, 15.0)

# Speed range
V_range = np.linspace(10, 20, 300)

# Fuel and time functions
def F(V): return a + b*V + c*V**2 + d*V**3
def Ds(V): return L / (24 * V)
def D(V): return Ds(V) + Dp
def R(V): return freight_rate * D(V)
def R_adj(V): return R(V) + (K * L / 24) * (1 / VR - 1 / V)

# Model equations
def Z1(V): return (R(V) - C * D(V) - F(V) * Fc * Ds(V)) / D(V)
def Z2(V): return (Ca + F(V) * Fc) * L / (24 * V)
def Z3(V): return (R_adj(V) - C * D(V) - F(V) * Fc * Ds(V)) / D(V)

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

# Output Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model 1 Speed", f"{V1_opt:.2f} kn")
    st.metric("Profit (Z1)", f"${Z1_opt:,.0f}")
with col2:
    st.metric("Model 2 Speed", f"{V2_opt:.2f} kn")
    st.metric("Cost (Z2)", f"${Z2_opt:,.0f}")
with col3:
    st.metric("Model 3 Speed", f"{V3_opt:.2f} kn")
    st.metric("Profit (Z3)", f"${Z3_opt:,.0f}")

# Create three charts with aligned y-axes
colA, colB, colC = st.columns(3)

with colA:
    st.markdown("### Model 1: Daily Profit")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_range, y=Z1_vals, name="Z1", line=dict(color="blue"),
                              hovertemplate="Speed: %{x:.2f} kn<br>Z1: %{y:,.0f}"))
    fig1.add_vline(x=V1_opt, line_dash="dash", line_color="blue")
    fig1.add_vline(x=assumed_speed, line_dash="dot", line_color="gray")
    fig1.update_layout(yaxis=dict(title="Z ($/day)", range=[min(Z2_vals)*0.95, max(Z2_vals)*1.05]),
                       xaxis_title="Speed (kn)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    st.markdown("### Model 2: Total Cost")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=V_range, y=Z2_vals, name="Z2", line=dict(color="orange"),
                              hovertemplate="Speed: %{x:.2f} kn<br>Z2: %{y:,.0f}"))
    fig2.add_vline(x=V2_opt, line_dash="dash", line_color="orange")
    fig2.add_vline(x=assumed_speed, line_dash="dot", line_color="gray")
    fig2.update_layout(yaxis=dict(title="Z ($)", range=[min(Z2_vals)*0.95, max(Z2_vals)*1.05]),
                       xaxis_title="Speed (kn)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with colC:
    st.markdown("### Model 3: Profit with Bonus")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=V_range, y=Z3_vals, name="Z3", line=dict(color="green"),
                              hovertemplate="Speed: %{x:.2f} kn<br>Z3: %{y:,.0f}"))
    fig3.add_vline(x=V3_opt, line_dash="dash", line_color="green")
    fig3.add_vline(x=assumed_speed, line_dash="dot", line_color="gray")
    fig3.update_layout(yaxis=dict(title="Z ($/day)", range=[min(Z2_vals)*0.95, max(Z2_vals)*1.05]),
                       xaxis_title="Speed (kn)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# Expandable S&C table
with st.expander("📋 Speed vs Fuel Consumption Table"):
    speeds = np.arange(10, 21)
    consumptions = [F(v) for v in speeds]
    df = pd.DataFrame({"Speed (knots)": speeds, "Fuel Consumption (tons/day)": consumptions})
    st.dataframe(df, use_container_width=True)

# Tooltip section
with st.expander("ℹ️ Model Equations & Explanation"):
    st.markdown("**Model 1:** Income-Generating Profit Optimization")
    st.latex(r"Z_1 = rac{R - C(D_s + D_p) - F(V) \cdot F_c \cdot D_s}{D_s + D_p}")
    st.markdown("Maximizes daily profit from cargo revenue.")

    st.markdown("**Model 2:** Ballast Leg Cost Minimization")
    st.latex(r"Z_2 = \left(C_a + F(V) \cdot F_c\right) \cdot \frac{L}{24V}")
ight) \cdot rac{L}{24V}")
    st.markdown("Minimizes cost of empty voyage using daily opportunity and fuel cost.")

    st.markdown("**Model 3:** Bonus/Penalty Adjustment")
    st.latex(r"R' = R + rac{K L}{24} \left(rac{1}{V_R} - rac{1}{V}
ight)")
    st.latex(r"Z_3 = rac{R' - C(D_s + D_p) - F(V) \cdot F_c \cdot D_s}{D_s + D_p}")
    st.markdown("Maximizes adjusted profit accounting for early arrival bonus or delay penalty.")
