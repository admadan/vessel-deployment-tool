import streamlit as st
import numpy as np
import plotly.graph_objects as go

# === Helper Functions ===
def calculate_freight_revenue(freight_rate_per_day, L, Dp, V):
    Ds = L / (24 * V)
    total_days = Ds + Dp
    return freight_rate_per_day * total_days, Ds, total_days

def fuel_at_speed(F0, V, V0):
    return F0 * (V / V0) ** 3

def model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C):
    return [
        (R - (C * (Dp + L / (24 * V)) + fuel_at_speed(F0, V, V0) * Fc * L / (24 * V))) / (Dp + L / (24 * V))
        for V in V_range
    ]

def model2_cost_curve(V_range, Ca, V0, F0, Fc, L):
    return [
        (Ca + fuel_at_speed(F0, V, V0) * Fc) * L / (24 * V)
        for V in V_range
    ]

def model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR):
    return [
        ((R + (K * L / 24) * (1 / VR - 1 / V)) -
         (C * (Dp + L / (24 * V)) + fuel_at_speed(F0, V, V0) * Fc * L / (24 * V))) / (Dp + L / (24 * V))
        for V in V_range
    ]

def find_optimum(V_range, Z_curve, mode='max'):
    Z_array = np.array(Z_curve)
    idx = np.argmax(Z_array) if mode == 'max' else np.argmin(Z_array)
    return V_range[idx], Z_array[idx]

def operating_cost(C, F, Fc, Ds):
    return (C + F * Fc) * Ds

# === UI Setup ===
st.set_page_config(page_title="Ronen Speed Optimization", layout="wide")
st.title("🚢 Ronen Optimal Speed Dashboard – Final Version")

# === Inputs ===
with st.sidebar:
    st.header("Input Parameters")
    
    L = st.number_input("Voyage Distance (nm)", value=4000, help="Total nautical miles to be sailed")
    Dp = st.number_input("Port Days", value=2.0, help="Number of days spent in port (loading/unloading)")
    st.markdown("### ℹ️ Speed Inputs Explained")
    st.markdown("- **V₀ (Reference Speed):** Used to estimate fuel consumption using cube law.")
    st.markdown("- **Vm (Minimum Speed):** Used as the lower bound in optimization.")
    st.markdown("- **Assumed Speed:** Used to calculate revenue from freight/day × voyage days.")
    st.markdown("- **VR (Reference Speed for Bonus):** Used in Model 3 to calculate bonus/penalty for early/late arrival.")
    
    V0 = st.slider("Reference Speed (V0) [knots]", 0.0, 25.0, 19.0, help="Speed at which fuel consumption F₀ is measured")
    F0 = st.slider("Main Engine Fuel/day at V0 (tons)", 50.0, 300.0, 120.0, help="Main engine fuel consumption at reference speed V₀")
    Fc = st.slider("Fuel Cost ($/ton)", 200, 1200, 800, help="Cost per ton of fuel consumed")
    C = st.slider("Daily Ops Cost ($)", 5000, 50000, 12000, help="Fixed daily operating cost, excluding propulsion")
    Vm = st.slider("Minimum Speed Vm [knots]", 0.0, 15.0, 10.0, help="Minimum allowed vessel speed during optimization")
    
    freight_rate = st.slider("Freight Rate ($/day)", 0, 200000, 100000, step=5000, help="Expected revenue per day from charterer")
    assumed_speed = st.slider("Assumed Speed for Revenue [knots]", 0.0, V0, 15.0, help="Used to calculate total voyage duration for revenue")
    
    Ca = st.slider("Alternative Value of Ship ($/day)", 20000, 100000, 70000, help="Daily value of the vessel if idle or under alternative use")
    K = st.slider("Bonus/Penalty per day ($)", 0, 50000, 25000, help="Value of bonus or penalty per day based on arrival deviation")
    VR = st.slider("Reference Contract Speed (VR) [knots]", 0.0, 25.0, 18.0, help="Contractual speed used in Model 3 bonus/penalty calculation")
    
    # === Calculations ===
    V_range = np.linspace(Vm, V0, 300)
    R, Ds_assumed, D_assumed = calculate_freight_revenue(freight_rate, L, Dp, assumed_speed)
    
    Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
    Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
    Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)
    
    V1_opt, Z1_opt = find_optimum(V_range, Z1)
    V2_opt, Z2_opt = find_optimum(V_range, Z2, mode='min')
    V3_opt, Z3_opt = find_optimum(V_range, Z3)
    
    # Model 1
    Ds1, D1 = L / (24 * V1_opt), Dp + L / (24 * V1_opt)
    F1 = fuel_at_speed(F0, V1_opt, V0)
    P1 = Z1_opt * D1
    OC1 = operating_cost(C, F1, Fc, Ds1)
    
    # Model 2
    Ds2, D2 = L / (24 * V2_opt), Dp + L / (24 * V2_opt)
    F2 = fuel_at_speed(F0, V2_opt, V0)
    OC2 = operating_cost(Ca, F2, Fc, Ds2)
    
    # Model 3
    Ds3, D3 = L / (24 * V3_opt), Dp + L / (24 * V3_opt)
    F3 = fuel_at_speed(F0, V3_opt, V0)
    P3 = Z3_opt * D3
    OC3 = operating_cost(C, F3, Fc, Ds3)
    
    # === Metrics and Charts ===
    col1, col2, col3 = st.columns(3)
    

col1, col2, col3 = st.columns(3)
with col1:
st.subheader("📘 Model 1: Fixed Revenue")
    st.markdown(f"- **Optimum Speed:** {V1_opt:.2f} kn")
    st.caption("Speed (V) affects:")
    
            st.markdown("""
    <sub>Speed (V) affects sea days \(D_s = \frac{L}{24V}\), fuel use \(F \propto V^3\), and voyage time \(D = D_s + D_p\)</sub>
    """, unsafe_allow_html=True)
    st.markdown(f"- **Daily Profit (Z):** ${Z1_opt:,.0f}")
    st.markdown(f"- **Total Profit:** ${P1:,.0f}")
    st.markdown(f"- **Total Op Cost:** ${OC1:,.0f}")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_range, y=Z1, name="Model 1", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=[V1_opt], y=[Z1_opt], mode='markers+text', name="Optimum", text=[f"{V1_opt:.2f} kn"], marker=dict(size=10, color='blue')))
    fig1.update_layout(title="Model 1: Daily Profit", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
st.subheader("📙 Model 2: Ballast Leg")
    st.markdown(f"- **Optimum Speed:** {V2_opt:.2f} kn")
    st.caption("Speed (V) affects:")
    
            st.markdown("""
    <sub>Speed (V) affects sea days \(D_s = \frac{L}{24V}\), fuel use \(F \propto V^3\), and voyage time \(D = D_s + D_p\)</sub>
    """, unsafe_allow_html=True)
    st.markdown(f"- **Total Cost (Z):** ${Z2_opt:,.0f}")
    st.markdown(f"- **Total Op Cost:** ${OC2:,.0f}")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=V_range, y=Z2, name="Model 2", line=dict(color='orange')))
    fig2.add_trace(go.Scatter(x=[V2_opt], y=[Z2_opt], mode='markers+text', name="Optimum", text=[f"{V2_opt:.2f} kn"], marker=dict(size=10, color='orange')))
    fig2.update_layout(title="Model 2: Total Cost", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
st.subheader("📗 Model 3: Bonus/Penalty")
    st.markdown(f"- **Optimum Speed:** {V3_opt:.2f} kn")
    st.caption("Speed (V) affects:")
    
            st.markdown("""
    <sub>Speed (V) affects sea days \(D_s = \frac{L}{24V}\), fuel use \(F \propto V^3\), and voyage time \(D = D_s + D_p\)</sub>
    """, unsafe_allow_html=True)
    st.markdown(f"- **Daily Profit (Z):** ${Z3_opt:,.0f}")
    st.markdown(f"- **Total Profit:** ${P3:,.0f}")
    st.markdown(f"- **Total Op Cost:** ${OC3:,.0f}")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=V_range, y=Z3, name="Model 3", line=dict(color='green')))
    fig3.add_trace(go.Scatter(x=[V3_opt], y=[Z3_opt], mode='markers+text', name="Optimum", text=[f"{V3_opt:.2f} kn"], marker=dict(size=10, color='green')))
    fig3.update_layout(title="Model 3: Profit with Bonus/Penalty", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
    
    # === Info Section ===

with st.expander("ℹ️ Model Equations"):
    st.markdown("### **Model 1 – Income-Generating**")
    st.markdown("**Daily Profit:**")
        st.markdown("### **Model 2 – Ballast (Empty Leg)**")
    st.markdown("**Total Cost:**")
        st.markdown("**Optimal Speed:**")
        st.markdown("### **Model 3 – Bonus/Penalty Contracts**")
    st.markdown("**Adjusted Revenue:**")
        st.markdown("**Daily Profit:**")

