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

    L = st.number_input("Voyage Distance (nm)", value=4000)
    V0_input = st.number_input("Speed used to measure ME Fuel/day (knots)", value=19.0)
    F0_input = st.number_input("Main Engine Fuel Consumption at V0 (tons/day)", value=120.0)
    gen_cons = st.number_input("Generator Consumption (tons/day)", value=5.0)
    Fc = st.number_input("Fuel Cost ($/ton)", value=800.0)
    Dp = st.number_input("Port Days", value=2.0)
    Vm = st.number_input("Minimum Speed Vm (knots)", value=10.0)

    C = gen_cons * Fc
    st.markdown(f"**💡 Daily Operating Cost C:** ${C:,.0f} (Gen cons × Fuel cost)")

    Ca = st.number_input("Alternative Value (Ca, $/day)", value=70000)
    freight_rate = st.slider("Freight Rate ($/day)", 0, 200000, 100000, 5000)
    assumed_speed = st.slider("Assumed Speed (knots)", Vm, V0_input, 15.0)

    K = st.number_input("Bonus/Penalty (K, $/day)", value=25000)
    VR = st.number_input("Reference Speed (VR)", value=18.0)

# === Calculations ===
V_range = np.linspace(Vm, V0_input, 300)
R, voyage_days = calculate_freight_revenue(freight_rate, L, Dp, assumed_speed)
F0 = F0_input
V0 = V0_input

Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)

V1_opt, Z1_opt = find_optimum(V_range, Z1)
V2_opt, Z2_opt = find_optimum(V_range, Z2, 'min')
V3_opt, Z3_opt = find_optimum(V_range, Z3)

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
if Z1_opt < Ca and Z3_opt < Ca:
    st.error("Both Model 1 and 3 generate less profit than the vessel's alternative value. Model 2 (cost minimization) is optimal.")
    st.info("Model 1 and Model 3 are less useful when no revenue justifies longer sailing or early arrival incentives.")
elif Z1_opt > Z3_opt:
    st.success(f"Model 1 is more profitable at {V1_opt:.2f} knots. Daily profit Z = ${Z1_opt:,.0f}")
    st.info("Model 3 is not optimal because the bonus/penalty adjustment didn't increase revenue enough.")
else:
    st.success(f"Model 3 is more profitable at {V3_opt:.2f} knots. Daily profit Z = ${Z3_opt:,.0f}")
    st.info("Model 1 was outperformed due to timing incentives that made early arrival more valuable.")

# === About Section ===
with st.expander("ℹ️ About the Ronen Models"):
    st.markdown(r'''
### Model 1: Income-Generating Leg
Used when the voyage earns a fixed total revenue \( R \).

**Daily Profit**:
$$
Z = \frac{R - C(D_s + D_p) - F \cdot F_c \cdot D_s}{D_s + D_p}
$$

---

### Model 2: Empty (Ballast) Leg
Used when vessel is repositioning and earns no freight.

**Total Cost**:
$$
Z = \left(C_a + F_0 \cdot F_c \cdot \left(\frac{V}{V_0}\right)^3\right) \cdot \frac{L}{24V}
$$

**Optimal Speed**:
$$
V^* = V_0 \cdot \left(\frac{C_a}{2F_0F_c}\right)^{1/3}
$$

---

### Model 3: Bonus/Penalty Contracts
Used when freight includes incentives for early or late arrival.

**Adjusted Revenue**:
$$
R' = R + \frac{K \cdot L}{24} \left(\frac{1}{V_R} - \frac{1}{V}\right)
$$

**Daily Profit**:
$$
Z = \frac{R' - C(D_s + D_p) - F \cdot F_c \cdot D_s}{D_s + D_p}
$$
    ''')
