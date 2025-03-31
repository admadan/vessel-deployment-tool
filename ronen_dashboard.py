
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# === Model Functions ===
def calculate_freight_revenue(freight_rate_per_day, L, Dp, V):
    Ds = L / (24 * V)
    total_days = Ds + Dp
    return freight_rate_per_day * total_days, total_days

def model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C):
    return [
        (R - (C * (Dp + L / (24 * V)) + F0 * (V / V0)**3 * Fc * L / (24 * V))) / (Dp + L / (24 * V))
        for V in V_range
    ]

def model2_cost_curve(V_range, Ca, V0, F0, Fc, L):
    return [
        (Ca + F0 * (V / V0)**3 * Fc) * L / (24 * V)
        for V in V_range
    ]

def model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR):
    return [
        (
            (R + (K * L / 24) * (1 / VR - 1 / V)) -
            (C * (Dp + L / (24 * V)) + F0 * (V / V0)**3 * Fc * L / (24 * V))
        ) / (Dp + L / (24 * V))
        for V in V_range
    ]

def find_optimum(V_range, Z_curve, mode='max'):
    Z_array = np.array(Z_curve)
    idx = np.argmax(Z_array) if mode == 'max' else np.argmin(Z_array)
    return V_range[idx], Z_array[idx]

# === Streamlit UI ===
st.set_page_config(page_title="Ronen Speed Models", layout="wide")
st.title("🚢 Ronen Optimal Speed Dashboard")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("User Inputs")

    model_choice = st.selectbox("Select Model", ["Model 1: Revenue", "Model 2: Empty Leg", "Model 3: Bonus/Penalty"])

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

# === Plotting Based on Model ===
fig = go.Figure()
y_axis_title = "Z ($/day)"

if "Model 1" in model_choice:
    Z = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
    V_opt, Z_opt = find_optimum(V_range, Z)
    fig.add_trace(go.Scatter(x=V_range, y=Z, name="Model 1: Daily Profit", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[V_opt], y=[Z_opt], mode='markers+text', name="Optimum", text=[f"{V_opt:.2f} kn"], marker=dict(size=10, color='blue')))
    y_axis_title = "Daily Profit ($)"

elif "Model 2" in model_choice:
    Z = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
    V_opt, Z_opt = find_optimum(V_range, Z, mode='min')
    fig.add_trace(go.Scatter(x=V_range, y=Z, name="Model 2: Total Cost", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=[V_opt], y=[Z_opt], mode='markers+text', name="Optimum", text=[f"{V_opt:.2f} kn"], marker=dict(size=10, color='orange')))
    y_axis_title = "Total Cost ($)"

elif "Model 3" in model_choice:
    Z = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)
    V_opt, Z_opt = find_optimum(V_range, Z)
    fig.add_trace(go.Scatter(x=V_range, y=Z, name="Model 3: Daily Profit (Adj)", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[V_opt], y=[Z_opt], mode='markers+text', name="Optimum", text=[f"{V_opt:.2f} kn"], marker=dict(size=10, color='green')))
    y_axis_title = "Daily Profit ($)"

# === Plot Layout ===
fig.update_layout(
    title=f"Z vs Speed – {model_choice}",
    xaxis_title="Speed (knots)",
    yaxis_title=y_axis_title,
    template="plotly_white",
    height=600,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
