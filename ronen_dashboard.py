
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
st.title("🚢 Ronen Optimal Speed Dashboard – All Models View")

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

# === Run All Models ===
Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)

V1_opt, Z1_opt = find_optimum(V_range, Z1)
V2_opt, Z2_opt = find_optimum(V_range, Z2, mode='min')
V3_opt, Z3_opt = find_optimum(V_range, Z3)

# === Determine Best Model ===
if Z1_opt < Ca and Z3_opt < Ca:
    best_model = "Model 2"
elif Z1_opt > Z3_opt:
    best_model = "Model 1"
else:
    best_model = "Model 3"

# === Chart ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=V_range, y=Z1, name="Model 1: Daily Profit", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=V_range, y=Z3, name="Model 3: Profit (Bonus/Penalty)", line=dict(color='green', dash='dash')))
fig.add_trace(go.Scatter(x=V_range, y=Z2, name="Model 2: Total Cost", line=dict(color='orange', dash='dot')))
fig.add_hline(y=Ca, line=dict(color='red', dash='dot'), annotation_text="Alternative Daily Value")

fig.add_trace(go.Scatter(x=[V1_opt], y=[Z1_opt], mode='markers+text', name="Model 1 Opt", text=[f"{V1_opt:.2f} kn"], marker=dict(size=10, color='blue')))
fig.add_trace(go.Scatter(x=[V2_opt], y=[Z2_opt], mode='markers+text', name="Model 2 Opt", text=[f"{V2_opt:.2f} kn"], marker=dict(size=10, color='orange')))
fig.add_trace(go.Scatter(x=[V3_opt], y=[Z3_opt], mode='markers+text', name="Model 3 Opt", text=[f"{V3_opt:.2f} kn"], marker=dict(size=10, color='green')))

fig.update_layout(title="📊 Daily Profit / Cost vs Speed", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white", height=600)
st.plotly_chart(fig, use_container_width=True)

# === Recommendations ===
st.subheader("💡 Recommendation")

if best_model == "Model 2":
    st.error("Both Model 1 and 3 generate less profit than the vessel's alternative value. Model 2 (cost minimization) is optimal.")
    st.info("Model 1 and 3 are less applicable because no revenue justifies longer sailing or early arrival.")
elif best_model == "Model 1":
    st.success(f"✅ Model 1 is the best choice at {V1_opt:.2f} knots. Daily profit Z = ${Z1_opt:,.0f}")
    st.info("Model 3 was not optimal due to limited benefit from timing contracts.")
else:
    st.success(f"✅ Model 3 is the best choice at {V3_opt:.2f} knots. Daily profit Z = ${Z3_opt:,.0f}")
    st.info("Model 1 was outperformed due to bonus/penalty incentives.")

