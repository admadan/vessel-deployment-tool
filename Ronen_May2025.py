
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar

st.set_page_config(page_title="Optimal Speed from Cubic Polynomial", layout="wide")
st.title("🚢 Optimal Speed from Polynomial Fuel Curve – Ronen Model 1")

# Sidebar Inputs
with st.sidebar:
    st.header("Fuel Curve Coefficients (F(V) = a + bV + cV² + dV³)")
    a = st.number_input("Coefficient a", value=15.0)
    b = st.number_input("Coefficient b", value=-1.5)
    c = st.number_input("Coefficient c", value=0.9)
    d = st.number_input("Coefficient d", value=0.015)

    st.header("Operational Inputs")
    Fc = st.number_input("Fuel Cost ($/ton)", value=800)
    C = st.number_input("Daily Operational Cost ($/day)", value=12000)
    L = st.number_input("Voyage Distance (nm)", value=4000)
    Dp = st.number_input("Port Days", value=2.0)
    freight_rate = st.number_input("Freight Rate ($/day)", value=100000)
    assumed_speed = st.slider("Assumed Speed (kn)", 10.0, 20.0, 15.0)

# Functions
def F(V): return a + b*V + c*V**2 + d*V**3
def Ds(V): return L / (24 * V)
def D(V): return Ds(V) + Dp
def R(V): return freight_rate * D(V)

def Z(V):
    try:
        return (R(V) - C * D(V) - F(V) * Fc * Ds(V)) / D(V)
    except ZeroDivisionError:
        return -np.inf

# Evaluate Z over range
V_range = np.linspace(10, 20, 300)
Z_values = [Z(v) for v in V_range]

# Optimize
result = minimize_scalar(lambda v: -Z(v), bounds=(10, 20), method='bounded')
V_opt = result.x
Z_opt = Z(V_opt)
Z_assumed = Z(assumed_speed)

# Layout
col1, col2 = st.columns(2)
with col1:
    st.metric("🔵 Optimum Speed", f"{V_opt:.2f} kn")
    st.metric("💰 Daily Profit (Zopt)", f"${Z_opt:,.0f}")
with col2:
    st.metric("📍 Assumed Speed", f"{assumed_speed:.2f} kn")
    st.metric("📉 Savings", f"${(Z_opt - Z_assumed) * D(V_opt):,.0f}")

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=V_range, y=Z_values, mode="lines", name="Daily Profit (Z)", line=dict(color="blue")))
fig.add_vline(x=V_opt, line_dash="dash", line_color="blue", annotation_text=f"Zopt: ${Z_opt:,.0f}")
fig.add_vline(x=assumed_speed, line_dash="dot", line_color="gray", annotation_text=f"Zassumed: ${Z_assumed:,.0f}")
fig.update_layout(title="Z vs Speed", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Explanation Section
with st.expander("ℹ️ Model Explanation"):
    st.markdown("""
    - This model uses **Model 1 (Income Leg)** from Ronen (1982).
    - The fuel consumption is defined by a **user-input cubic polynomial**:  
      \( F(V) = a + bV + cV^2 + dV^3 \)
    - The profit function is:  
      \[
      Z(V) = \frac{R - C(D_s + D_p) - F(V) F_c D_s}{D_s + D_p}
      \]
    - Where:  
        - \( D_s = \frac{L}{24V} \) is sea days  
        - \( D_p \) is port days  
        - \( R = \text{freight rate} \times (D_s + D_p) \)
    """)
