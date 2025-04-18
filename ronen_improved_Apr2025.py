
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# === Vessel Polynomial Fuel Curve (mocked per vessel) ===
vessels = {
    "LNG Vessel A": {"coeffs": [15.0, -1.8, 0.9, 0.015], "EF": 2.75},
    "LNG Vessel B": {"coeffs": [18.0, -1.5, 1.1, 0.012], "EF": 2.75}
}

# === UI Setup ===
st.set_page_config(page_title="Ronen Fuel Curve + ETS", layout="wide")
st.title("🚢 Ronen Model – Polynomial Fuel Curve + ETS")

# Sidebar
with st.sidebar:
    st.header("Input Parameters")
    vessel_name = st.selectbox("Select Vessel", list(vessels.keys()))
    a, b, c, d = vessels[vessel_name]["coeffs"]
    EF = vessels[vessel_name]["EF"]

    L = st.number_input("Voyage Distance (nm)", value=4000)
    Dp = st.number_input("Port Days", value=2.0)
    Vmin = st.slider("Min Speed (Vm)", 10.0, 17.0, 12.0)
    Vmax = st.slider("Max Speed (V0)", 17.0, 25.0, 19.0)
    assumed_speed = st.slider("Assumed Speed (knots)", Vmin, Vmax, 15.0)

    C = st.slider("Daily Ops Cost ($)", 5000, 50000, 12000)
    Fc = st.slider("Fuel Cost ($/ton)", 300, 1200, 800)

    CO2_price = st.slider("CO₂ Price ($/tCO₂)", 50, 200, 100)
    EU_share = st.slider("EU ETS Exposure (%)", 0, 100, 50)

    freight_rate = st.slider("Freight Rate ($/day)", 0, 200000, 100000, step=5000)
    Ca = st.number_input("Alternative Vessel Value ($/day)", value=70000)
    K = st.slider("Bonus/Penalty Rate ($)", 0, 50000, 25000)
    VR = st.slider("Reference Contract Speed (VR)", 10.0, 25.0, 18.0)

# === Helper Functions ===
def F(V): return a + b*V + c*V**2 + d*V**3
def Ds(V): return L / (24 * V)
def D(V): return Ds(V) + Dp
def ETS_cost(V): return F(V) * EF * CO2_price * (EU_share / 100)

def R_total(V): return freight_rate * D(V)

def model1(V):
    return (R_total(V) - (C * D(V) + F(V) * (Fc + ETS_cost(V)) * Ds(V))) / D(V)

def model2(V):
    return (Ca + F(V) * Fc) * L / (24 * V)

def model3(V):
    R_adj = R_total(V) + (K * L / 24) * (1 / VR - 1 / V)
    return (R_adj - (C * D(V) + F(V) * (Fc + ETS_cost(V)) * Ds(V))) / D(V)

# === Calculations ===
V_range = np.linspace(Vmin, Vmax, 300)
Z1 = [model1(V) for V in V_range]
Z2 = [model2(V) for V in V_range]
Z3 = [model3(V) for V in V_range]

def find_opt(Z, mode='max'):
    idx = np.argmax(Z) if mode == 'max' else np.argmin(Z)
    return V_range[idx], Z[idx]

V1_opt, Z1_opt = find_opt(Z1)
V2_opt, Z2_opt = find_opt(Z2, mode='min')
V3_opt, Z3_opt = find_opt(Z3)

Z1_assumed = model1(assumed_speed)
Z2_assumed = model2(assumed_speed)
Z3_assumed = model3(assumed_speed)

# === Output Display ===
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📘 Model 1: Income Leg")
    Ds1 = Ds(V1_opt)
    OC1 = (C + F(V1_opt) * Fc) * Ds1
    P1 = Z1_opt * (Ds1 + Dp)
    st.markdown(f"- **Optimum Speed:** {V1_opt:.2f} kn")
    st.markdown(f"- **Daily Profit:** ${Z1_opt:,.0f}")
    st.markdown(f"- **Total Profit:** ${P1:,.0f}")
    st.markdown(f"- **% Improvement:** {(Z1_opt - Z1_assumed)/Z1_assumed*100:.2f}%")
    st.markdown(f"- **Savings:** ${(Z1_opt - Z1_assumed) * (Dp + Ds1):,.0f}")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_range, y=Z1, name="Model 1"))
    fig1.add_vline(x=V1_opt, line_dash='dash', line_color='blue')
    fig1.add_annotation(x=V1_opt, y=Z1_opt, text=f"Zopt: ${Z1_opt:,.0f}", showarrow=True)
    fig1.add_annotation(x=assumed_speed, y=Z1_assumed, text=f"Zassumed: ${Z1_assumed:,.0f}", showarrow=True)
    fig1.update_layout(title="Model 1: Daily Profit", xaxis_title="Speed (kn)", yaxis_title="Z ($/day)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📙 Model 2: Ballast Leg")
    Ds2 = Ds(V2_opt)
    OC2 = (Ca + F(V2_opt) * Fc) * Ds2
    st.markdown(f"- **Optimum Speed:** {V2_opt:.2f} kn")
    st.markdown(f"- **Total Cost:** ${Z2_opt:,.0f}")
    st.markdown(f"- **% Cost Reduction:** {(Z2_assumed - Z2_opt)/Z2_assumed*100:.2f}%")
    st.markdown(f"- **Savings:** ${(Z2_assumed - Z2_opt):,.0f}")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=V_range, y=Z2, name="Model 2"))
    fig2.add_vline(x=V2_opt, line_dash='dash', line_color='orange')
    fig2.add_annotation(x=V2_opt, y=Z2_opt, text=f"Zopt: ${Z2_opt:,.0f}", showarrow=True)
    fig2.add_annotation(x=assumed_speed, y=Z2_assumed, text=f"Zassumed: ${Z2_assumed:,.0f}", showarrow=True)
    fig2.update_layout(title="Model 2: Total Cost", xaxis_title="Speed (kn)", yaxis_title="Z ($)")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.subheader("📗 Model 3: Bonus/Penalty")
    Ds3 = Ds(V3_opt)
    P3 = Z3_opt * (Ds3 + Dp)
    st.markdown(f"- **Optimum Speed:** {V3_opt:.2f} kn")
    st.markdown(f"- **Daily Profit:** ${Z3_opt:,.0f}")
    st.markdown(f"- **Total Profit:** ${P3:,.0f}")
    st.markdown(f"- **% Improvement:** {(Z3_opt - Z3_assumed)/Z3_assumed*100:.2f}%")
    st.markdown(f"- **Savings:** ${(Z3_opt - Z3_assumed) * (Dp + Ds3):,.0f}")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=V_range, y=Z3, name="Model 3"))
    fig3.add_vline(x=V3_opt, line_dash='dash', line_color='green')
    fig3.add_annotation(x=V3_opt, y=Z3_opt, text=f"Zopt: ${Z3_opt:,.0f}", showarrow=True)
    fig3.add_annotation(x=assumed_speed, y=Z3_assumed, text=f"Zassumed: ${Z3_assumed:,.0f}", showarrow=True)
    fig3.update_layout(title="Model 3: Profit (Bonus)", xaxis_title="Speed (kn)", yaxis_title="Z ($/day)")
    st.plotly_chart(fig3, use_container_width=True)

# === Info Section ===
with st.expander("ℹ️ Model Equations and Savings"):
    st.markdown("### **Model 1 – Income-Generating**")
    st.latex(r"Z = \frac{R - C(D_s + D_p) - F \cdot (F_c + ETS) \cdot D_s}{D_s + D_p}")

    st.markdown("### **Model 2 – Ballast (Empty Leg)**")
    st.latex(r"Z = \left(C_a + F \cdot F_c\r\right) \cdot \frac{L}{24V}")
\right) \cdot \frac{L}{24V}")
    st.latex(r"V^* = V_0 \left(\frac{C_a}{2 F_0 F_c}
ight)^{1/3} 	ext{ (from Ronen, for cube law)}")

    st.markdown("### **Model 3 – Bonus/Penalty Contracts**")
    st.latex(r"R' = R + \frac{K L}{24} \left(\frac{1}{V_R} - \frac{1}{V}
ight)")
    st.latex(r"Z = \frac{R' - C(D_s + D_p) - F \cdot (F_c + ETS) \cdot D_s}{D_s + D_p}")

    st.markdown("### 💡 **Savings Logic**")
    st.latex(r"	ext{Savings} = (Z_{	ext{opt}} - Z_{	ext{assumed}}) \cdot (D_s + D_p)")
    st.latex(r"	ext{Cost Reduction} = Z_{	ext{assumed}} - Z_{	ext{opt}}")
