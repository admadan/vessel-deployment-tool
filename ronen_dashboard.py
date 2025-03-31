import streamlit as st
import numpy as np
import plotly.graph_objects as go

# === Model Functions ===
def calculate_freight_revenue(freight_rate_per_day, L, Dp, V):
    Ds = L / (24 * V)
    total_days = Ds + Dp
    return freight_rate_per_day * total_days, total_days

def model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C):
    return [(R - (C * (Dp + L / (24 * V)) + F0 * (V / V0)**3 * Fc * L / (24 * V))) / (Dp + L / (24 * V)) for V in V_range]

def model2_cost_curve(V_range, Ca, V0, F0, Fc, L):
    return [(Ca + F0 * (V / V0)**3 * Fc) * L / (24 * V) for V in V_range]

def model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR):
    return [((R + (K * L / 24) * (1 / VR - 1 / V)) -
             (C * (Dp + L / (24 * V)) + F0 * (V / V0)**3 * Fc * L / (24 * V))) / (Dp + L / (24 * V)) for V in V_range]

def find_optimum(V_range, Z_curve, mode='max'):
    Z_array = np.array(Z_curve)
    idx = np.argmax(Z_array) if mode == 'max' else np.argmin(Z_array)
    return V_range[idx], Z_array[idx]

# === Streamlit UI ===
st.set_page_config(page_title="Ronen Speed Models", layout="wide")
st.title("🚢 Ronen Optimal Speed Dashboard – Tab View")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("User Inputs")

    L = st.number_input("Voyage Distance (nm)", value=4000)
    V0 = st.number_input("Speed used to measure ME Fuel/day (knots)", value=19.0)
    F0 = st.number_input("Main Engine Fuel Consumption at V0 (tons/day)", value=120.0)
    gen_cons = st.number_input("Generator Consumption (tons/day)", value=5.0)
    Fc = st.number_input("Fuel Cost ($/ton)", value=800.0)
    Dp = st.number_input("Port Days", value=2.0)
    Vm = st.number_input("Minimum Speed Vm (knots)", value=10.0)
    C = gen_cons * Fc
    st.markdown(f"**💡 Daily Operating Cost C:** ${C:,.0f}")

    freight_rate = st.slider("Freight Rate ($/day)", 0, 200000, 100000, 5000)
    assumed_speed = st.slider("Assumed Speed (knots)", Vm, V0, 15.0)
    R, voyage_days = calculate_freight_revenue(freight_rate, L, Dp, assumed_speed)

    Ca = st.number_input("Alternative Value (Ca, $/day)", value=70000)
    K = st.number_input("Bonus/Penalty (K, $/day)", value=25000)
    VR = st.number_input("Reference Speed (VR)", value=18.0)

# === Calculation Range ===
V_range = np.linspace(Vm, V0, 300)

# === Tab Layout ===
tab1, tab2, tab3 = st.tabs(["📘 Model 1", "📙 Model 2", "📗 Model 3"])

# --- Model 1 ---
with tab1:
    Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
    V1_opt, Z1_opt = find_optimum(V_range, Z1)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_range, y=Z1, name="Daily Profit", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=[V1_opt], y=[Z1_opt], mode='markers+text', name="Optimum", text=[f"{V1_opt:.2f} kn"], marker=dict(size=10, color='blue')))
    fig1.update_layout(title="Model 1: Daily Profit vs Speed", xaxis_title="Speed (knots)", yaxis_title="Daily Profit ($)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)
    st.info(f"Optimal speed: {V1_opt:.2f} knots | Daily profit Z = ${Z1_opt:,.0f}")

# --- Model 2 ---
with tab2:
    Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
    V2_opt, Z2_opt = find_optimum(V_range, Z2, mode='min')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=V_range, y=Z2, name="Total Cost", line=dict(color='orange')))
    fig2.add_trace(go.Scatter(x=[V2_opt], y=[Z2_opt], mode='markers+text', name="Optimum", text=[f"{V2_opt:.2f} kn"], marker=dict(size=10, color='orange')))
    fig2.update_layout(title="Model 2: Total Cost vs Speed", xaxis_title="Speed (knots)", yaxis_title="Total Cost ($)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    st.info(f"Optimal speed: {V2_opt:.2f} knots | Total voyage cost Z = ${Z2_opt:,.0f}")

# --- Model 3 ---
with tab3:
    Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)
    V3_opt, Z3_opt = find_optimum(V_range, Z3)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=V_range, y=Z3, name="Daily Profit (Adj)", line=dict(color='green')))
    fig3.add_trace(go.Scatter(x=[V3_opt], y=[Z3_opt], mode='markers+text', name="Optimum", text=[f"{V3_opt:.2f} kn"], marker=dict(size=10, color='green')))
    fig3.update_layout(title="Model 3: Profit with Bonus/Penalty vs Speed", xaxis_title="Speed (knots)", yaxis_title="Daily Profit ($)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
    st.info(f"Optimal speed: {V3_opt:.2f} knots | Daily profit Z = ${Z3_opt:,.0f}")
