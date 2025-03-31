
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# === Model Functions ===
def calculate_freight_revenue(freight_rate_per_day, L, Dp, V):
    Ds = L / (24 * V)
    total_days = Ds + Dp
    return freight_rate_per_day * total_days, Ds, total_days

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
st.title("🚢 Ronen Optimal Speed Dashboard – Metrics & Charts")

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
    R, Ds_input, D_input = calculate_freight_revenue(freight_rate, L, Dp, assumed_speed)

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

Ds1, D1 = L / (24 * V1_opt), Dp + L / (24 * V1_opt)
Ds2, D2 = L / (24 * V2_opt), Dp + L / (24 * V2_opt)
Ds3, D3 = L / (24 * V3_opt), Dp + L / (24 * V3_opt)

P1 = Z1_opt * D1
P2 = Z2_opt
P3 = Z3_opt * D3

# === GLOBAL METRICS ===
st.markdown("### 🌍 Global Voyage Metrics")
st.markdown(f"- Assumed Speed: **{assumed_speed:.2f} knots**")
st.markdown(f"- Voyage Days: **{D_input:.2f}**")
st.markdown(f"- Sea Days: **{Ds_input:.2f}**")
st.markdown(f"- Total Freight Revenue: **${R:,.0f}**")

# === 3 Columns with Charts and Metrics ===
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📘 Model 1")
    st.markdown(f"- Optimum Speed: **{V1_opt:.2f} kn**")
    st.markdown(f"- Daily Profit (Z): **${Z1_opt:,.0f}**")
    st.markdown(f"- Total Profit: **${P1:,.0f}**")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_range, y=Z1, name="Model 1", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=[V1_opt], y=[Z1_opt], mode='markers+text', name="Optimum", text=[f"{V1_opt:.2f} kn"], marker=dict(size=10, color='blue')))
    fig1.update_layout(title="Model 1: Daily Profit", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📙 Model 2")
    st.markdown(f"- Optimum Speed: **{V2_opt:.2f} kn**")
    st.markdown(f"- Total Cost: **${Z2_opt:,.0f}**")
    st.markdown(f"- Voyage Days: **{D2:.2f}**")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=V_range, y=Z2, name="Model 2", line=dict(color='orange')))
    fig2.add_trace(go.Scatter(x=[V2_opt], y=[Z2_opt], mode='markers+text', name="Optimum", text=[f"{V2_opt:.2f} kn"], marker=dict(size=10, color='orange')))
    fig2.update_layout(title="Model 2: Total Cost", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.subheader("📗 Model 3")
    st.markdown(f"- Optimum Speed: **{V3_opt:.2f} kn**")
    st.markdown(f"- Daily Profit (Z): **${Z3_opt:,.0f}**")
    st.markdown(f"- Total Profit: **${P3:,.0f}**")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=V_range, y=Z3, name="Model 3", line=dict(color='green')))
    fig3.add_trace(go.Scatter(x=[V3_opt], y=[Z3_opt], mode='markers+text', name="Optimum", text=[f"{V3_opt:.2f} kn"], marker=dict(size=10, color='green')))
    fig3.update_layout(title="Model 3: Profit with Bonus/Penalty", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# === Recommendation Section ===
st.subheader("📌 Recommendation")

model_scores = {
    "Model 1": P1 if Z1_opt >= Ca else -1e9,
    "Model 2": -P2,
    "Model 3": P3 if Z3_opt >= Ca else -1e9
}
best_model = max(model_scores, key=model_scores.get)

if best_model == "Model 2":
    st.error("⚠️ Both Model 1 and Model 3 generate less profit than the vessel's alternative value. Model 2 (cost minimization) is optimal.")
else:
    st.success(f"✅ {best_model} is the most profitable under the given voyage conditions.")
