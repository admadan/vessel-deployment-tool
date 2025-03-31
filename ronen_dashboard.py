
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
st.title("🚢 Ronen Optimal Speed Dashboard – Compare All Models")

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

Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)

V1_opt, Z1_opt = find_optimum(V_range, Z1)
V2_opt, Z2_opt = find_optimum(V_range, Z2, mode='min')
V3_opt, Z3_opt = find_optimum(V_range, Z3)

# === Plotting All Three Models ===
col1, col2, col3 = st.columns(3)

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_range, y=Z1, name="Daily Profit", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=[V1_opt], y=[Z1_opt], mode='markers+text', name="Optimum", text=[f"{V1_opt:.2f} kn"], marker=dict(size=10, color='blue')))
    fig1.update_layout(title="Model 1: Daily Profit", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=V_range, y=Z2, name="Total Cost", line=dict(color='orange')))
    fig2.add_trace(go.Scatter(x=[V2_opt], y=[Z2_opt], mode='markers+text', name="Optimum", text=[f"{V2_opt:.2f} kn"], marker=dict(size=10, color='orange')))
    fig2.update_layout(title="Model 2: Total Cost", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=V_range, y=Z3, name="Daily Profit (Adj)", line=dict(color='green')))
    fig3.add_trace(go.Scatter(x=[V3_opt], y=[Z3_opt], mode='markers+text', name="Optimum", text=[f"{V3_opt:.2f} kn"], marker=dict(size=10, color='green')))
    fig3.update_layout(title="Model 3: Bonus/Penalty", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# === Recommendation Section ===
st.subheader("📌 Recommendation")

model_scores = {
    "Model 1": Z1_opt if Z1_opt >= Ca else -1e9,
    "Model 2": -Z2_opt,  # Lower is better
    "Model 3": Z3_opt if Z3_opt >= Ca else -1e9
}
best_model = max(model_scores, key=model_scores.get)

if best_model == "Model 2":
    st.error("⚠️ Both Model 1 and Model 3 produce less profit than the vessel's alternative value. Model 2 (cost minimization) is recommended.")
else:
    st.success(f"✅ {best_model} is the optimal choice based on the input conditions.")
    if best_model == "Model 1":
        st.info("Model 1 provides the highest daily profit and should be used when consistent freight revenue is guaranteed.")
    elif best_model == "Model 3":
        st.info("Model 3 offers better profitability due to arrival-based bonuses or penalties, ideal for speed-linked contracts.")
