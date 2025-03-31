import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Model Calculation Functions ---
def calculate_freight_revenue(freight_rate_per_day, L, Dp, V):
    sea_days = L / (24 * V)
    total_days = sea_days + Dp
    revenue = freight_rate_per_day * total_days
    return revenue, total_days

def model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C):
    return [
        (R - (C * (Dp + L / (24 * V)) + F0 * (V / V0) ** 3 * Fc * L / (24 * V))) / (Dp + L / (24 * V))
        for V in V_range
    ]

def model2_cost_curve(V_range, Ca, V0, F0, Fc, L):
    return [
        (Ca + F0 * (V / V0) ** 3 * Fc) * L / (24 * V)
        for V in V_range
    ]

def model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR):
    return [
        (
            (R + (K * L / 24) * (1 / VR - 1 / V)) -
            (C * (Dp + L / (24 * V)) + F0 * (V / V0) ** 3 * Fc * L / (24 * V))
        ) / (Dp + L / (24 * V))
        for V in V_range
    ]

def find_optimum(V_range, Z_curve, mode='max'):
    Z_array = np.array(Z_curve)
    idx = np.argmax(Z_array) if mode == 'max' else np.argmin(Z_array)
    return V_range[idx], Z_array[idx]

# --- Streamlit App Layout ---
st.set_page_config(page_title="Ronen Speed Models (Plotly)", layout="wide")
st.title("🚢 Ronen Optimal Speed Dashboard (Interactive with Plotly)")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📌 Input Parameters")
    L = st.number_input("Voyage Distance (nm)", value=4000)
    V0 = st.number_input("Nominal Speed (knots)", value=19.0)
    F0 = st.number_input("Fuel Consumption @ V0 (tons/day)", value=120.0)
    Fc = st.number_input("Fuel Cost ($/ton)", value=800.0)
    Dp = st.number_input("Port Days", value=2)
    C = st.number_input("Operating Cost ($/day)", value=12000.0)
    Ca = st.number_input("Alternative Value ($/day)", value=70000.0)

    st.markdown("### 💰 Revenue Inputs")
    freight_rate = st.slider("Freight Rate ($/day)", min_value=0, max_value=200000, value=100000, step=5000, format="%,d")
    assumed_speed = st.slider("Assumed Speed (knots)", 10.0, V0, 15.0)

    st.markdown("### 📦 Model 3 (Timing Contracts)")
    K = st.number_input("Penalty/Bonus ($/day late/early)", value=25000)
    VR = st.number_input("Reference Speed (knots)", value=18.0)

# --- Calculate Revenue from Freight Rate ---
R, voyage_days = calculate_freight_revenue(freight_rate, L, Dp, assumed_speed)
st.markdown(f"""
### 📈 Freight Revenue: **${R:,.0f}**
- Based on freight rate **${freight_rate:,.0f}/day** and voyage duration **{voyage_days:.2f} days**
""")

# --- Model Calculations ---
V_range = np.linspace(10, V0, 300)
Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)

V1_opt, Z1_opt = find_optimum(V_range, Z1, 'max')
V2_opt, Z2_opt = find_optimum(V_range, Z2, 'min')
V3_opt, Z3_opt = find_optimum(V_range, Z3, 'max')

# --- Plotly Chart ---
fig = go.Figure()

fig.add_trace(go.Scatter(x=V_range, y=Z1, mode='lines', name='Model 1: Daily Profit', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=V_range, y=Z3, mode='lines', name='Model 3: Daily Profit (Bonus/Penalty)', line=dict(color='green', dash='dash')))
fig.add_trace(go.Scatter(x=V_range, y=Z2, mode='lines', name='Model 2: Total Cost', line=dict(color='orange', dash='dot')))
fig.add_hline(y=Ca, line=dict(color='red', dash='dot'), annotation_text="Alternative Daily Value", annotation_position="top left")

fig.add_trace(go.Scatter(x=[V1_opt], y=[Z1_opt], mode='markers+text', name='Model 1 Opt', text=[f"{V1_opt:.2f} kn"], textposition="top center", marker=dict(size=10, color='blue')))
fig.add_trace(go.Scatter(x=[V3_opt], y=[Z3_opt], mode='markers+text', name='Model 3 Opt', text=[f"{V3_opt:.2f} kn"], textposition="top center", marker=dict(size=10, color='green')))
fig.add_trace(go.Scatter(x=[V2_opt], y=[Z2_opt], mode='markers+text', name='Model 2 Opt', text=[f"{V2_opt:.2f} kn"], textposition="top center", marker=dict(size=10, color='orange')))

fig.update_layout(
    title="📊 Daily Profit / Cost vs Speed (Ronen Models)",
    xaxis_title="Speed (knots)",
    yaxis_title="Daily Profit or Cost ($)",
    template='plotly_white',
    hovermode='x unified',
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --- Business Recommendation ---
st.subheader("💡 Recommendation")
if Z1_opt < Ca and Z3_opt < Ca:
    st.error("⚠️ Daily profit from both Model 1 and Model 3 is less than the ship's alternative value. Use Model 2 (cost minimization).")
elif Z1_opt > Z3_opt:
    st.success(f"✅ Model 1 is more profitable at **{V1_opt:.2f} knots** with daily profit **${Z1_opt:,.0f}**.")
else:
    st.success(f"✅ Model 3 (with timing contract) is more profitable at **{V3_opt:.2f} knots** with daily profit **${Z3_opt:,.0f}**.")

st.markdown("Use the sidebar to test different freight rates, fuel prices, and contract terms. The chart updates dynamically.")



with st.expander("ℹ️ About Ronen’s Speed Optimization Models"):
    st.markdown("""
### **Model 1: Income-Generating Leg**

Used when the vessel generates **fixed freight revenue** per voyage.

**Objective:** Maximize daily profit.

**Daily profit formula**:
\\[
Z = \\frac{R - C(D_s + D_p) - F \\cdot F_c \\cdot D_s}{D_s + D_p}
\\]

Where:
- \\( R \\): total voyage revenue ($)  
- \\( D_s = \\frac{L}{24V} \\): sea days  
- \\( D_p \\): port days  
- \\( C \\): daily vessel operating cost ($)  
- \\( F = F_0 \\left(\\frac{V}{V_0}\\right)^3 \\): fuel use at speed V  
- \\( F_c \\): fuel price ($/ton)  
- \\( V_0 \\): nominal speed  
- \\( F_0 \\): fuel consumption at nominal speed

**Optimization:** Solve cubic equation for \\( V \\) that maximizes \\( Z \\)

---

### **Model 2: Empty (Positioning) Leg**

Used when there's **no revenue**, and the vessel is repositioning.

**Objective:** Minimize total voyage cost.

**Total cost formula**:
\\[
Z = D_s \\cdot C_a + F \\cdot F_c \\cdot D_s = \\left(C_a + F_0 \\cdot F_c \\cdot \\left(\\frac{V}{V_0}\\right)^3\\right) \\cdot \\frac{L}{24V}
\\]

Where:
- \\( C_a \\): alternative daily value of ship (opportunity cost)

**Optimal speed:**
\\[
V^* = V_0 \\left(\\frac{C_a}{2F_0F_c}\\right)^{1/3}
\\]

---

### **Model 3: Bonus/Penalty Contracts**

Used when contracts reward early delivery or penalize delay.

**Adjusted revenue**:
\\[
R' = R + \\frac{K \\cdot L}{24} \\left(\\frac{1}{V_R} - \\frac{1}{V}\\right)
\\]

**Daily profit becomes:**
\\[
Z = \\frac{R' - C(D_s + D_p) - F \\cdot F_c \\cdot D_s}{D_s + D_p}
\\]

Where:
- \\( K \\): penalty (if late) or bonus (if early) per day  
- \\( V_R \\): reference contractual speed  
- \\( R' \\): adjusted revenue

**Optimization:** Solve cubic equation for \\( V \\) again.

---

These models are derived from:  
📘 *Ronen, D. (1982). The effect of oil price on the optimal speed of ships. Journal of the Operational Research Society, 33(11), 1035–1040.*
    """)
