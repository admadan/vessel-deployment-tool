import streamlit as st
import numpy as np
import os
import random
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import string
from functools import partial

# Hide Streamlit's default menu and GitHub link
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)



# Inject custom CSS to style the buttons
st.markdown(
    """
    <style>
    div.stButton > button {
        width: 100%; /* Make buttons take full width of their container */
        padding: 10px; /* Add some padding inside the buttons */
        font-size: 16px; /* Increase font size */
    }
    div.css-1r6slb0 { /* Target the top right corner element */
        position: absolute;
        top: 10px;
        right: 10px;
    }
    div.css-1r6slb0 button {
        width: auto !important; /* Override full width for this button */
    }
    # .st-emotion-cache-1y4p8pa { /* Adjust margin between columns */
    #     gap: 1em;
    # }

    .top-right-button-container {
        position: absolute;
        width: 10%;
        top: 10px;
        right: 10px;
        z-index: 1; /* Ensure button is above other elements if needed */
    }
    .top-right-button-container button {
        width: auto !important;  /* Override full width */
    }
    </style>
    </style>
    """,
    unsafe_allow_html=True,
)



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




def optimal_speed_calculation_page():
    st.title("Optimal Speed Calculation")
    col_button_empty, col_button = st.columns([5, 1])
    with col_button:
        if st.button("Go Back to Main"):
            st.session_state.page = "main"
    # === Inputs ===
    with st.sidebar:
        st.header("Input Parameters")
        L = st.number_input("Voyage Distance (nm)", value=4000)
        Dp = st.number_input("Port Days", value=2.0)
        V0 = st.slider("Reference Speed (V0) [knots]", 0.0, 25.0, 19.0)
        F0 = st.slider("Main Engine Fuel/day at V0 (tons)", 50.0, 300.0, 120.0)
        Fc = st.slider("Fuel Cost ($/ton)", 200, 1200, 800)
        C = st.slider("Daily Ops Cost ($)", 5000, 50000, 12000)
        Vm = st.slider("Minimum Speed Vm [knots]", 0.0, 15.0, 10.0)
        freight_rate = st.slider("Freight Rate ($/day)", 0, 200000, 100000, step=5000)
        assumed_speed = st.slider("Assumed Speed for Revenue [knots]", 0.0, V0, 15.0)
        Ca = st.slider("Alternative Value of Ship ($/day)", 20000, 100000, 70000)
        K = st.slider("Bonus/Penalty per day ($)", 0, 50000, 25000)
        VR = st.slider("Reference Contract Speed (VR) [knots]", 0.0, 25.0, 18.0)

    # === Calculations ===
    V_range = np.linspace(Vm, V0, 300)
    R, Ds_assumed, D_assumed = calculate_freight_revenue(freight_rate, L, Dp, assumed_speed)

    Z1 = model1_profit_curve(V_range, R, L, Dp, V0, F0, Fc, C)
    Z2 = model2_cost_curve(V_range, Ca, V0, F0, Fc, L)
    Z3 = model3_profit_curve(V_range, R, K, L, Dp, V0, F0, Fc, C, VR)

    V1_opt, Z1_opt = find_optimum(V_range, Z1)
    V2_opt, Z2_opt = find_optimum(V_range, Z2, mode='min')
    V3_opt, Z3_opt = find_optimum(V_range, Z3)

    # Assumed Z values for improvement
    Z1_assumed = model1_profit_curve([assumed_speed], R, L, Dp, V0, F0, Fc, C)[0]
    Z2_assumed = model2_cost_curve([assumed_speed], Ca, V0, F0, Fc, L)[0]
    Z3_assumed = model3_profit_curve([assumed_speed], R, K, L, Dp, V0, F0, Fc, C, VR)[0]

    # === Results ===
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.subheader("📘 Model 1: Fixed Revenue")
        Ds1 = L / (24 * V1_opt)
        F1 = fuel_at_speed(F0, V1_opt, V0)
        OC1 = operating_cost(C, F1, Fc, Ds1)
        P1 = Z1_opt * (Ds1 + Dp)
        st.markdown(f"- **Optimum Speed:** {V1_opt:.2f} kn")
        st.markdown(f"- **Daily Profit (Z):** ${Z1_opt:,.0f}")
        st.markdown(f"- **Total Profit:** ${P1:,.0f}")
        st.markdown(f"- **Total Op Cost:** ${OC1:,.0f}")
        st.markdown(f"- **% Improvement:** {(Z1_opt - Z1_assumed)/Z1_assumed*100:.2f}%")
        st.markdown(f"- **Savings:** ${(Z1_opt - Z1_assumed) * (Dp + Ds1):,.0f}")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=V_range, y=Z1, name="Model 1", line=dict(color='blue')))
        fig1.add_vline(x=V1_opt, line_dash='dash', line_color='blue')

        fig1.add_vline(x=V1_opt, line_dash='dash', line_color='blue')
        fig1.add_annotation(x=V1_opt, y=Z1_opt, text=f"Zopt: ${Z1_opt:,.0f}", showarrow=True, arrowhead=1)
        fig1.add_annotation(x=assumed_speed, y=Z1_assumed, text=f"Zassumed: ${Z1_assumed:,.0f}", showarrow=True, arrowhead=2)
        fig1.update_layout(title="Model 1: Daily Profit", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("📙 Model 2: Ballast Leg")
        Ds2 = L / (24 * V2_opt)
        F2 = fuel_at_speed(F0, V2_opt, V0)
        OC2 = operating_cost(Ca, F2, Fc, Ds2)
        st.markdown(f"- **Optimum Speed:** {V2_opt:.2f} kn")
        st.markdown(f"- **Total Cost (Z):** ${Z2_opt:,.0f}")
        st.markdown(f"- **Total Op Cost:** ${OC2:,.0f}")
        st.markdown(f"- **% Cost Reduction:** {(Z2_assumed - Z2_opt)/Z2_assumed*100:.2f}%")
        st.markdown(f"- **Savings:** ${(Z2_assumed - Z2_opt):,.0f}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=V_range, y=Z2, name="Model 2", line=dict(color='orange')))
        fig2.add_vline(x=V2_opt, line_dash='dash', line_color='orange')

        fig2.add_vline(x=V2_opt, line_dash='dash', line_color='orange')
        fig2.add_annotation(x=V2_opt, y=Z2_opt, text=f"Zopt: ${Z2_opt:,.0f}", showarrow=True, arrowhead=1)
        fig2.add_annotation(x=assumed_speed, y=Z2_assumed, text=f"Zassumed: ${Z2_assumed:,.0f}", showarrow=True, arrowhead=2)
        fig2.update_layout(title="Model 2: Total Cost", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.subheader("📗 Model 3: Bonus/Penalty")
        Ds3 = L / (24 * V3_opt)
        F3 = fuel_at_speed(F0, V3_opt, V0)
        OC3 = operating_cost(C, F3, Fc, Ds3)
        P3 = Z3_opt * (Ds3 + Dp)
        st.markdown(f"- **Optimum Speed:** {V3_opt:.2f} kn")
        st.markdown(f"- **Daily Profit (Z):** ${Z3_opt:,.0f}")
        st.markdown(f"- **Total Profit:** ${P3:,.0f}")
        st.markdown(f"- **Total Op Cost:** ${OC3:,.0f}")
        st.markdown(f"- **% Improvement:** {(Z3_opt - Z3_assumed)/Z3_assumed*100:.2f}%")
        st.markdown(f"- **Savings:** ${(Z3_opt - Z3_assumed) * (Dp + Ds3):,.0f}")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=V_range, y=Z3, name="Model 3", line=dict(color='green')))
        fig3.add_vline(x=V3_opt, line_dash='dash', line_color='green')

        fig3.add_vline(x=V3_opt, line_dash='dash', line_color='green')
        fig3.add_annotation(x=V3_opt, y=Z3_opt, text=f"Zopt: ${Z3_opt:,.0f}", showarrow=True, arrowhead=1)
        fig3.add_annotation(x=assumed_speed, y=Z3_assumed, text=f"Zassumed: ${Z3_assumed:,.0f}", showarrow=True, arrowhead=2)
        fig3.update_layout(title="Model 3: Profit with Bonus/Penalty", xaxis_title="Speed (knots)", yaxis_title="Z ($/day)", template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    # === Info Section ===

    with st.expander("ℹ️ Model Equations"):
        st.markdown("### **Model 1 – Income-Generating**")
        st.markdown("**Daily Profit:**")
        st.latex(r"Z = \frac{R - C(D_s + D_p) - F \cdot F_c \cdot D_s}{D_s + D_p}")

        st.markdown("### **Model 2 – Ballast (Empty Leg)**")
        st.markdown("**Total Cost:**")
        st.latex(r"Z = \left(C_a + F_0 F_c \left(\frac{V}{V_0}\right)^3\right) \cdot \frac{L}{24V}")
        st.markdown("**Optimal Speed:**")
        st.latex(r"V^* = V_0 \left(\frac{C_a}{2 F_0 F_c}\right)^{1/3}")

        st.markdown("### **Model 3 – Bonus/Penalty Contracts**")
        st.markdown("**Adjusted Revenue:**")
        st.latex(r"R' = R + \frac{K L}{24} \left(\frac{1}{V_R} - \frac{1}{V}\right)")
        st.markdown("**Daily Profit:**")
        st.latex(r"Z = \frac{R' - C(D_s + D_p) - F \cdot F_c \cdot D_s}{D_s + D_p}")

        st.markdown("### 💡 Savings Logic")
        st.markdown("**Model 1 & 3 (Profit):**")
        st.latex(r"\text{Savings} = (Z_{\text{opt}} - Z_{\text{assumed}}) \times (D_s + D_p)")
        st.markdown("**Model 2 (Cost):**")
        st.latex(r"\text{Savings} = Z_{\text{assumed}} - Z_{\text{opt}}")

def get_value(key, default):
    """Helper function to get values from st.session_state or return default."""
    return st.session_state.get(key, default)


if __name__ == "__main__":
    import streamlit as st
    def main_page():
        st.title("Main Navigation")
        col1, col2= st.columns(2)
        with col1:
            if st.button("Vessel Input Section"):
                st.session_state.page = "page_1"
            if st.button("Deployment Simulation"):
                st.session_state.page = "page_2"
            if st.button("Voyage Simulation"):
                st.session_state.page = "page_3"
        with col2:

            if st.button("Feright Rate Monitoring"):
                st.session_state.page = "page_5_content"
                
            if st.button("Market Condition"):
                st.session_state.page = "page_4"
        
            
            if st.button("Optimal Speed Calculation"):
                st.session_state.page = "page_6"
                
    def add_back_button():
        col_button_empty, col_button = st.columns([5, 1])
        with col_button:
            if st.button("Go Back to Main"):
                st.session_state.page = "main"
   


    
    def page_1():
        st.header("🛠 Vessel Input Section")
    
     # Retrieve num_vessels from session_state if it exists, otherwise set default to 10
        if "num_vessels" not in st.session_state:
            st.session_state["num_vessels"] = 10  # Default value
    
        # User input for number of vessels
        num_vessels = st.number_input("How many LNG Carriers to display?", 
                                      min_value=1, 
                                      max_value=26, 
                                      value=st.session_state["num_vessels"])
        
        # Update the session state with the new value of num_vessels
        st.session_state["num_vessels"] = num_vessels
        
        # Generate vessel names like A, B, C, ...
        vessel_names = [f"LNG Carrier {chr(65 + i)}" for i in range(num_vessels)]
        
        # Load or initialize vessel_data
        if "vessel_data" not in st.session_state:
            if os.path.exists("data/vessel_data.csv"):
                st.session_state["vessel_data"] = pd.read_csv("data/vessel_data.csv")
            else:
                # Create initial vessel data if the CSV does not exist
                vessel_names = [f"LNG Carrier {chr(65 + i)}" for i in range(num_vessels)]
                st.session_state["vessel_data"] = pd.DataFrame({
                    "Vessel_ID": range(1, num_vessels + 1),
                    "Name": vessel_names,
                    "Capacity_CBM": [160000] * num_vessels,
                    "FuelEU_GHG_Compliance": ([65, 80, 95] * ((num_vessels // 3) + 1))[:num_vessels],
                    "CII_Rating": (["A", "B", "C"] * ((num_vessels // 3) + 1))[:num_vessels],
                    "Boil_Off_Rate_percent": ([0.08, 0.09, 0.07] * ((num_vessels // 3) + 1))[:num_vessels],
                    "Margin": [2000] * num_vessels,
                    "Operational_m": [50000] * num_vessels,
                    "Performance_Profile": ["good"] * num_vessels,
                    "Actual_GHG_Intensity": [50] * num_vessels
                })
    
        # Adjust vessel_data if new vessels added or removed
        vessel_data = st.session_state["vessel_data"]
        
        if len(vessel_data) < num_vessels:
            diff = num_vessels - len(vessel_data)
            new_data = pd.DataFrame({
                "Vessel_ID": range(len(vessel_data) + 1, num_vessels + 1),
                "Name": [f"LNG Carrier {chr(65 + i)}" for i in range(len(vessel_data), num_vessels)],
                "Capacity_CBM": [160000] * diff,
                "FuelEU_GHG_Compliance": ([65, 80, 95] * ((diff // 3) + 1))[:diff],
                "CII_Rating": (["A", "B", "C"] * ((diff // 3) + 1))[:diff],
                "Boil_Off_Rate_percent": ([0.08, 0.09, 0.07] * ((diff // 3) + 1))[:diff],
                "Margin": [2000] * diff,
                "Operational_m": [50000] * diff,
                "Performance_Profile": ["good"] * diff,
                "Actual_GHG_Intensity": [50] * diff
            })
            st.session_state["vessel_data"] = pd.concat([vessel_data, new_data], ignore_index=True)
            vessel_data = st.session_state["vessel_data"]
    
        elif len(vessel_data) > num_vessels:
            vessel_data = vessel_data.iloc[:num_vessels]
            st.session_state["vessel_data"] = vessel_data
    
        # Save to CSV
        vessel_data.to_csv("data/vessel_data.csv", index=False)

         # ✅ Preload all performance tables for each vessel
        for i in range(num_vessels):
            session_key = f"editor_data_{i}"
            csv_path = f"data/vessel_{i}.csv"
            if session_key not in st.session_state:
                if os.path.exists(csv_path):
                    st.session_state[session_key] = pd.read_csv(csv_path)
                else:
                    st.session_state[session_key] = pd.DataFrame({
                        "Speed (knots)": [12.0, 14.0, 16.0],
                        "Fuel Consumption (tons/day)": [50.0, 70.0, 90.0]
                    })

    
        # Go Back button
        col_button_empty, col_button = st.columns([5, 1])
        with col_button:
            if st.button("Go Back to Main", key="back_to_main_page1"):
                st.session_state.page = "main"
    
        
        cols = st.columns(2)
        for idx in range(len(vessel_data)):
            with cols[idx % 2].expander(f"🚢 {vessel_data.at[idx, 'Name']}"):
                name = st.text_input("Vessel Name", value=vessel_data.at[idx, "Name"], key=f"name_{idx}")
                operational = st.number_input("Operational Cost (USD/day)", value=vessel_data.at[idx, "Operational_m"], key=f"operational_{idx}")
                margin = st.number_input("Margin (USD/day)", value=vessel_data.at[idx, "Margin"], key=f"margin_{idx}")
                ghg = st.number_input("Actual GHG Intensity (gCO2e/MJ)", value=vessel_data.at[idx, "Actual_GHG_Intensity"], key=f"ghg_{idx}")
    
                # Update in session_state
                vessel_data.at[idx, "Name"] = name
                vessel_data.at[idx, "Operational_m"] = operational
                vessel_data.at[idx, "Margin"] = margin
                vessel_data.at[idx, "Actual_GHG_Intensity"] = ghg
    
                # Speed vs Fuel Table Editor
                show_details = st.toggle("Show Performance Details", key=f"toggle_{idx}")
                if show_details:
                    st.subheader("✏ Speed vs. Fuel Consumption (tons/day)")
                    editor_key = f"editor_widget_{idx}"
                    session_key = f"editor_data_{idx}"
                    csv_path = f"data/vessel_{idx}.csv"
                # # After initializing vessel details (in page_1)
                # for i in range(num_vessels):
                #     table_key = f"editor_data_{i}"
                #     if table_key not in st.session_state:
                #         # Default performance data
                #         default_data = pd.DataFrame({
                #             "Speed (knots)": [12.0, 14.0, 16.0],
                #             "Fuel Consumption (tons/day)": [50.0, 70.0, 90.0]
                #         })
                #         st.session_state[table_key] = default_data
                    
    
                    if not os.path.exists("data"):
                        os.makedirs("data")
    
                    if session_key not in st.session_state:
                        if os.path.exists(csv_path):
                            st.session_state[session_key] = pd.read_csv(csv_path)
                        else:
                            st.session_state[session_key] = default_data.copy()
    
                    edited_data = st.data_editor(st.session_state[session_key], key=editor_key, num_rows="dynamic")
                    edited_data["Fuel Consumption (tons/day)"] = edited_data["Fuel Consumption (tons/day)"].round(1)
                    st.session_state[session_key] = edited_data
                    edited_data.to_csv(csv_path, index=False)
    
                    try:
                        speeds = edited_data["Speed (knots)"].dropna().astype(float).values
                        consumptions = edited_data["Fuel Consumption (tons/day)"].dropna().astype(float).values
    
                        if len(speeds) >= 2:
                            coeffs = np.polyfit(speeds, consumptions, 3)
                            a, b, c, d = coeffs[::-1]
    
                            st.markdown("### 📈 Fitted Cubic Curve Coefficients:")
                            st.markdown(f"**a** = {a:.3f}, **b** = {b:.3f}, **c** = {c:.3f}, **d** = {d:.5f}")
    
                            smooth_speeds = np.linspace(min(speeds), max(speeds), 100)
                            fitted_curve = a + b * smooth_speeds + c * smooth_speeds**2 + d * smooth_speeds**3
    
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=speeds, y=consumptions, mode='markers', name='User Input'))
                            fig.add_trace(go.Scatter(x=smooth_speeds, y=fitted_curve, mode='lines', name='Fitted Curve', line=dict(color='green')))
    
                            compare_toggle = st.checkbox("Compare with another vessel", key=f"compare_toggle_{idx}")
                            if compare_toggle:
                                compare_vessel = st.selectbox("Select vessel to compare", [v for i, v in enumerate(vessel_data['Name']) if i != idx], key=f"compare_{idx}")
                                compare_idx = vessel_data[vessel_data["Name"] == compare_vessel].index[0]
                                compare_data_path = f"data/vessel_{compare_idx}.csv"
    
                                if os.path.exists(compare_data_path):
                                    comp_data = pd.read_csv(compare_data_path)
                                    comp_speeds = comp_data["Speed (knots)"].dropna().astype(float).values
                                    comp_consumptions = comp_data["Fuel Consumption (tons/day)"].dropna().astype(float).values
    
                                    if len(comp_speeds) >= 3:
                                        comp_coeffs = np.polyfit(comp_speeds, comp_consumptions, 3)
                                        a2, b2, c2, d2 = comp_coeffs[::-1]
    
                                        comp_smooth = np.linspace(min(comp_speeds), max(comp_speeds), 100)
                                        comp_curve = a2 + b2 * comp_smooth + c2 * comp_smooth**2 + d2 * comp_smooth**3
    
                                        fig.add_trace(go.Scatter(
                                            x=comp_smooth,
                                            y=comp_curve,
                                            mode='lines',
                                            name=f"{compare_vessel} (Fitted)",
                                            line=dict(color='red', dash='dot')
                                        ))
                                    else:
                                        st.warning(f"Not enough data for comparison with {compare_vessel}.")
                                else:
                                    st.warning(f"No saved data found for {compare_vessel}.")
    
                            fig.update_layout(
                                title="Speed vs. Fuel Consumption",
                                xaxis_title="Speed (knots)",
                                yaxis_title="Fuel Consumption (tons/day)",
                                legend=dict(x=0.01, y=0.99)
                            )
                            st.plotly_chart(fig, use_container_width=True)
    
                    except Exception as e:
                        st.error(f"Error processing data: {e}")
    
        # Save vessel_data after edits
        st.session_state["vessel_data"] = vessel_data
        vessel_data.to_csv("data/vessel_data.csv", index=False)


    def page_2():
        st.title("Deployment Simulation")
    
        col_button_empty, col_button = st.columns([5, 1])
        with col_button:
            if st.button("Go Back to Main"):
                st.session_state.page = "main"
    
        # Use vessel_data from session state if available
        if "vessel_data" in st.session_state:
            vessel_data = st.session_state["vessel_data"]
        else:
            st.error("Vessel data not found. Please fill out data in the previous page first.")
            return
    
        col_input, col_results = st.columns([0.7, 4.2])  # Adjusted column widths
    
        with col_input:
            st.subheader("Simulation Inputs")
    
            # Use vessel_0.csv for available speeds (assumption)
            default_vessel_path = "data/vessel_0.csv"
            if os.path.exists(default_vessel_path):
                default_csv_data = pd.read_csv(default_vessel_path)
                speed_options = default_csv_data["Speed (knots)"].dropna().unique()
                selected_speed = st.selectbox("Select Speed (knots)", speed_options)
            else:
                st.warning("Speed options not found. Make sure data/vessel_0.csv exists.")
                return
    
            carbon_calc_method = st.selectbox("Carbon Calculation Method", ["Fixed Rate", "Boil Off Rate"])
            ets_price = st.number_input("ETS Price (USD/MT CO2)", value=75)
            lng_bunker_price = st.number_input("LNG Bunker Price (USD/MT)", value=600)
            required_ghg_intensity = st.number_input("Required GHG Intensity (gCO2e/MJ)", value=50)
            penalty_per_excess_unit = st.number_input("Penalty per Excess GHG Unit (USD)", value=1000)
            base_spot_rate = st.number_input("Base Spot Rate (USD/day)", value=120000)
    
        with col_results:
            st.subheader("Deployment Simulation Results")
            results = []
    
            for index, vessel in vessel_data.iterrows():
                vessel_data_path = f"data/vessel_{index}.csv"
    
                # Default fuel consumption
                fuel_consumption = 0
    
                if os.path.exists(vessel_data_path):
                    vessel_csv_data = pd.read_csv(vessel_data_path)
                    matching_row = vessel_csv_data[vessel_csv_data["Speed (knots)"] == selected_speed]
                    if not matching_row.empty:
                        fuel_consumption = matching_row["Fuel Consumption (tons/day)"].values[0]
    
                # Use Boil Off Rate if selected
                if carbon_calc_method == "Boil Off Rate":
                    adjusted_fuel = vessel["Boil_Off_Rate_percent"] * vessel["Capacity_CBM"] / 1000
                else:
                    adjusted_fuel = fuel_consumption
    
                auto_co2 = adjusted_fuel * 3.114
                carbon_cost = adjusted_fuel * ets_price
                fuel_cost = adjusted_fuel * lng_bunker_price
                margin_cost = vessel["Margin"]
    
                # GHG Penalty calculation
                ghg_penalty = 0
                if vessel["Actual_GHG_Intensity"] > required_ghg_intensity:
                    excess = vessel["Actual_GHG_Intensity"] - required_ghg_intensity
                    ghg_penalty = excess * penalty_per_excess_unit
                breakeven = fuel_cost + carbon_cost + margin_cost + ghg_penalty+ vessel["Operational_m"]
                vessel_market = base_spot_rate / breakeven if breakeven else 0
    
                results.append({
                    "Vessel": vessel["Name"],
                    "Fuel Cost ($/day)": f"{fuel_cost:,.1f}",
                    "Carbon Cost ($/day)": f"{carbon_cost:,.1f}",
                    "GHG Penalty ($/day)": f"{ghg_penalty:,.1f}",
                    "Margin ($/day)": f"{margin_cost:,.1f}",
                    "Operation Cost ($/day)": f"{vessel['Operational_m']:,.1f}",
                    # "Operation Cost ($/day)": f"{operation_m:,.1f}",
                    "Breakeven Spot ($/day)": f"{breakeven:,.1f}",
                    "Vessel Market": f"{vessel_market:.2f}",
                    "Decision": "Spot Recommended" if base_spot_rate > breakeven else "TC/Idle Preferred"
                })
    
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

    def page_3():
        st.title("Voyage Simulation Advisor")
    
        # ── Navigation ────────────────────────────────────────────────────────────────
        _, col = st.columns([5,1])
        with col:
            if st.button("Go Back to Main"):
                st.session_state.page = "main"
    
        # ── Preconditions ─────────────────────────────────────────────────────────────
        if "num_vessels" not in st.session_state:
            st.warning("Please enter the number of LNG carriers on Page 1 first.")
            return
    
        if "vessel_data" not in st.session_state:
            st.warning("No vessel data found. Please fill in Page 1 first.")
            return
    
        # ── Inputs ────────────────────────────────────────────────────────────────────
        num_vessels = st.session_state["num_vessels"]
        p             = st.number_input("Fuel Price (USD/MT)", value=700)
        ets_price     = st.number_input("ETS Price (USD/MT of CO2 eq.)", value=80)
        voyage_dist   = st.number_input("Voyage Distance (nm)", value=5000)
        freight_rate  = st.number_input("Freight Rate (USD/day)", value=60000)
    
        # ── Prepare vessel list from Page 1 ───────────────────────────────────────────
        vessel_data = st.session_state["vessel_data"].head(num_vessels)
    
        # ── Simulation per vessel ────────────────────────────────────────────────────
        for idx, vessel in vessel_data.iterrows():
            with st.expander(f"🛳️ {vessel['Name']} Voyage Simulation"):
    
                # Load the Speed vs Fuel table you edited on Page 1
                table_key = f"editor_data_{idx}"
                perf_df = st.session_state.get(table_key)
    
                if perf_df is None:
                    st.warning("No performance table found for this vessel.")
                    continue
    
                # Pre‐compute common factors
                op_cost_per_day = vessel["Operational_m"]
                margin          = vessel["Margin"]
    
                # Build the sim results using your exact speed rows
                sim_rows = []
                for _, row in perf_df.iterrows():
                    speed = float(row["Speed (knots)"])
                    fuel  = float(row["Fuel Consumption (tons/day)"])
                    days  = voyage_dist / (speed * 24)
                    fuel_cost        = fuel * p * days
                    ets_cost         = fuel * 3.114 * ets_price * days
                    operational_cost = op_cost_per_day * days
                    total_cost       = fuel_cost + ets_cost + operational_cost + margin * days
                    tce = (freight_rate * days - total_cost) / days
    
                    sim_rows.append({
                        "Speed (knots)": speed,
                        "Voyage Days": round(days, 1),
                        "Fuel Consumption (MT/day)": fuel,
                        "Fuel Cost ($)": round(fuel_cost, 1),
                        "ETS Cost ($)": round(ets_cost, 1),
                        "Operational Cost ($)": round(operational_cost, 1),
                        "Total Cost ($)": round(total_cost, 1),
                        "TCE ($/day)": round(tce, 1)
                    })
    
                sim_df = pd.DataFrame(sim_rows)
    
                # Display the data without highlighting
                st.dataframe(sim_df)
                best_idx = sim_df["TCE ($/day)"].idxmax()
                best = sim_df.loc[best_idx]
                st.success(
                    f"Optimal: {best['Speed (knots)']} knots → TCE ${best['TCE ($/day)']}/day"
                )


    
    def page_4():
        st.title("Market Condition")
    
        col_button_empty, col_button = st.columns([5, 1])
        with col_button:
            if st.button("Go Back to Main", key="market_condition_back_button"): # Added unique key
                st.session_state.page = "main"
    
        # Create two columns for layout
        col_inputs, col_results = st.columns(2)
    
        with col_inputs:
            st.subheader("Market Inputs")
            
            # Replacing sliders with input boxes
            st.session_state.fleet_size_number_supply = st.text_input("Fleet Size (# of Ships)", value=st.session_state.get('fleet_size_number_supply', 3131), key="fleet_size_number_supply_input")
            st.session_state.fleet_size_dwt_supply_in_dwt_million = st.text_input("Supply (DWT Million)", value=st.session_state.get('fleet_size_dwt_supply_in_dwt_million', 254.1), key="fleet_size_dwt_supply_in_dwt_million_input")
            st.session_state.utilization_constant = st.text_input("Utilization Factor", value=st.session_state.get('utilization_constant', 0.95), key="utilization_constant_input")
            st.session_state.assumed_speed = st.text_input("Speed (knots)", value=st.session_state.get('assumed_speed', 11.0), key="assumed_speed_input")
            st.session_state.sea_margin = st.text_input("Sea Margin (%)", value=st.session_state.get('sea_margin', 0.05), key="sea_margin_input")
            st.session_state.assumed_laden_days = st.text_input("Laden Days Fraction", value=st.session_state.get('assumed_laden_days', 0.4), key="assumed_laden_days_input")
            st.session_state.demand_billion_ton_mile = st.text_input("Demand (Bn Ton Mile)", value=st.session_state.get('demand_billion_ton_mile', 10396.0), key="demand_billion_ton_mile_input")
            st.session_state.auto_tightness = st.checkbox("Auto-calculate market tightness", value=st.session_state.get('auto_tightness', True), key="auto_tightness_checkbox")
            if not st.session_state.auto_tightness:
                st.session_state.manual_tightness = st.text_input("Manual Market Tightness", value=st.session_state.get("manual_tightness", 0.5), key="manual_tightness_input")
            st.session_state.base_spot_rate = st.text_input("Spot Rate (USD/day)", value=st.session_state.get('base_spot_rate', 60000), key="base_spot_rate_input")
            st.session_state.base_tc_rate = st.text_input("TC Rate (USD/day)", value=st.session_state.get('base_tc_rate', 50000), key="base_tc_rate_input")
    
            # st.session_state.carbon_calc_method = st.radio("Carbon Cost Based On", ["Main Engine Consumption", "Boil Off Rate"], index=["Main Engine Consumption", "Boil Off Rate"].index(st.session_state.get('carbon_calc_method', 'Main Engine Consumption')), key="carbon_calc_method_radio")
    
        with col_results:
            st.subheader("Equilibrium Calculations")
            
            try:
                fleet_size_number_supply = float(st.session_state.fleet_size_number_supply)
                fleet_size_dwt_supply_in_dwt_million = float(st.session_state.fleet_size_dwt_supply_in_dwt_million)
                utilization_constant = float(st.session_state.utilization_constant)
                assumed_speed = float(st.session_state.assumed_speed)
                sea_margin = float(st.session_state.sea_margin)
                assumed_laden_days = float(st.session_state.assumed_laden_days)
                demand_billion_ton_mile = float(st.session_state.demand_billion_ton_mile)
    
                dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
                distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
                productive_laden_days_per_year = assumed_laden_days * 365
                maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
                equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile
    
                if st.session_state.auto_tightness:
                    market_tightness = min(max(0.3 + (equilibrium / demand_billion_ton_mile), 0.0), 1.0)
                else:
                    market_tightness = float(st.session_state.get("manual_tightness", 0.5))
    
                sensitivity = abs(equilibrium / demand_billion_ton_mile)
    
                # Adding new fields for productivity and excess/deficit vessels
                productivity_per_vessel = maximum_supply_billion_ton_mile / fleet_size_number_supply
                excess_deficit_vessels = equilibrium / productivity_per_vessel
    
                # Displaying results
                st.write(f"DWT Utilization: {dwt_utilization:.1f} MT")
                st.write(f"Max Supply: {maximum_supply_billion_ton_mile:.1f} Bn Ton Mile")
                st.write(f"Equilibrium: {equilibrium:.1f} Bn Ton Mile")
                st.write(f"Market Condition: { 'Excess Supply' if equilibrium < 0 else 'Excess Demand'}")
                st.write(f"Market Tightness: {market_tightness:.2f}")
                st.write(f"Market Sensitivity: {sensitivity:.2%}")
                
                # New fields
                st.write(f"Productivity per Vessel: {productivity_per_vessel:.2f} Bn Ton Mile")
                st.write(f"Excess/Deficit Vessels: {excess_deficit_vessels:.1f} vessels")
            
            except ValueError:
                st.error("Please enter valid numeric values for the inputs.")




    def page_5_content(): # Renamed function to avoid conflict
        st.title("Feright Rate Monitoring")
        col_button_empty, col_button = st.columns([5, 1])
        with col_button:
            if st.button("Go Back to Main"):
                st.session_state.page = "main"
    
        st.title("📈 LNG Market Trends")
    
        base_url = "https://docs.google.com/spreadsheets/d/1kySjcfv1jMkDRrqAD9qS10KjIs5H1Vdu/gviz/tq?tqx=out:csv&sheet="
        sheet_names = {
        "Weekly": "Weekly%20data_160K%20CBM",
        "Monthly": "Monthly%20data_160K%20CBM",
        "Yearly": "Yearly%20data_160%20CBM"
    }

        freq_option = st.radio("Select Data Frequency", ["Weekly", "Monthly", "Yearly"])
        google_sheets_url = f"{base_url}{sheet_names[freq_option]}"
    
        df_filtered = pd.DataFrame()
    
        try:
            df_selected = pd.read_csv(google_sheets_url, dtype=str)
    
            if "Date" in df_selected.columns:
                df_selected["Date"] = pd.to_datetime(df_selected["Date"], errors='coerce')
                df_selected = df_selected.dropna(subset=["Date"]).sort_values(by="Date")
            else:
                st.error("⚠ 'Date' column not found in the dataset.")
    
            for col in df_selected.columns:
                if col != "Date":
                    df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0)
    
            available_columns = [col for col in df_selected.columns if col != "Date"]
            column_options = st.multiselect("Select Data Columns", available_columns, default=available_columns[:2] if available_columns else [])
    
            if "Date" in df_selected.columns:
                start_date = st.date_input("Select Start Date", df_selected["Date"].min())
                end_date = st.date_input("Select End Date", df_selected["Date"].max())
                df_filtered = df_selected[(df_selected["Date"] >= pd.to_datetime(start_date)) & (df_selected["Date"] <= pd.to_datetime(end_date))]
    
    
                if len(column_options) > 0:
                    num_plots = (len(column_options) + 1) // 2
                    specs = [[{'secondary_y': True}] for _ in range(num_plots)]
                    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.3, specs=specs)
    
                    for i in range(0, len(column_options), 2):
                        row_num = (i // 2) + 1
    
                        # Plot the first column in the pair
                        fig.add_trace(
                            go.Scatter(
                                x=df_filtered["Date"],
                                y=df_filtered[column_options[i]],
                                mode='lines',
                                name=column_options[i],
                                hovertemplate='Date: %{x}<br>Value: %{y}<extra></extra>',
                                showlegend=True,
                                legendgroup=column_options[i],
                            ),
                            row=row_num,
                            col=1,
                            secondary_y=False,
                        )
                        fig.update_yaxes(title_text=column_options[i], row=row_num, col=1, secondary_y=False)
    
                        # Plot the second column in the pair (if it exists)
                        if i + 1 < len(column_options):
                            fig.add_trace(
                                go.Scatter(
                                    x=df_filtered["Date"],
                                    y=df_filtered[column_options[i + 1]],
                                    mode='lines',
                                    name=column_options[i + 1],
                                    hovertemplate='Date: %{x}<br>Value: %{y}<extra></extra>',
                                    showlegend=True,
                                    legendgroup=column_options[i + 1],
                                ),
                                row=row_num,
                                col=1,
                                secondary_y=True,
                            )
                            fig.update_yaxes(title_text=column_options[i + 1], row=row_num, col=1, secondary_y=True)
    
    
                    fig.update_layout(
                        title="LNG Market Rates Over Time",
                        xaxis=dict(
                            title=f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                            tickangle=45,
                            tickformatstops=[dict(dtickrange=[None, None], value="%Y")],
                            range=[df_filtered["Date"].min(), df_filtered["Date"].max()]
                        ),
                        hovermode="x unified",
                        showlegend=True,  # Set showlegend to False at the layout level
                        height=300 * num_plots,
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
                else:
                    st.warning("Please select at least one data column.")
    
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")

    def page_6():
            
                optimal_speed_calculation_page()
           
                
# Define functions for the other pages (page_3, page_4, page_5, page_6) similarly

if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "page_1":
    page_1()
elif st.session_state.page == "page_2":
    page_2()
# Add elif blocks for the other pages
elif st.session_state.page == "page_3":
    page_3() # You'll need to define this function
elif st.session_state.page == "page_4":
    page_4() # You'll need to define this function
elif st.session_state.page == "page_5_content":
    page_5_content() # You'll need to define this function
elif st.session_state.page == "page_6":
    page_6() # You'll need to define this function
