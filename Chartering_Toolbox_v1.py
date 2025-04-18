import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
            st.title("Vessel Input Section")
            col_button_empty, col_button = st.columns([5, 1])
            with col_button:
                if st.button("Go Back to Main"):
                    st.session_state.page = "main"
            cols = st.columns(2)
            vessel_data = pd.DataFrame({
                "Vessel_ID": range(1, 11),
                "Name": [f"LNG Carrier {chr(65 + i)}" for i in range(10)],
                "Length_m": [295] * 10,
                "Beam_m": [46] * 10,
                "Draft_m": [11.5] * 10,
                "Capacity_CBM": [160000] * 10,
                "FuelEU_GHG_Compliance": [65, 65, 65, 80, 80, 80, 95, 95, 95, 95],
                "CII_Rating": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
                "Main_Engine_Consumption_MT_per_day": [70, 72, 74, 85, 88, 90, 100, 102, 105, 107],
                "Generator_Consumption_MT_per_day": [5, 5, 5, 6, 6, 6, 7, 7, 7, 7],
                "Boil_Off_Rate_percent": [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.07, 0.07, 0.07, 0.07],
                "Margin": [2000] * 10
            })
            for idx, row in vessel_data.iterrows():
                with cols[idx % 2].expander(f"🚢 {row['Name']}"):
                    vessel_data.at[idx, "Name"] = st.text_input("Vessel Name", value=row["Name"], key=f"name_{idx}")
                    vessel_data.at[idx, "Length_m"] = st.number_input("Length (m)", value=row["Length_m"], key=f"len_{idx}")
                    vessel_data.at[idx, "Beam_m"] = st.number_input("Beam (m)", value=row["Beam_m"], key=f"beam_{idx}")
                    vessel_data.at[idx, "Draft_m"] = st.number_input("Draft (m)", value=row["Draft_m"], key=f"draft_{idx}")
                    vessel_data.at[idx, "Margin"] = st.number_input("Margin (USD/day)", value=row["Margin"], key=f"margin_{idx}")
    
                    show_details = st.toggle("Show Performance Details", key=f"toggle_{idx}")
                    if show_details:
                        base_speed = 15
                        min_speed = max(8, base_speed - 3)
                        max_speed = min(20, base_speed + 3)
                        speed_range = list(range(int(min_speed), int(max_speed) + 1))
                        ref_total_consumption = row["Main_Engine_Consumption_MT_per_day"] + row[
                            "Generator_Consumption_MT_per_day"]
                        total_consumption = [ref_total_consumption * (speed / base_speed) ** 3 for speed in
                                             speed_range]
                        df_curve = pd.DataFrame(
                            {"Speed (knots)": speed_range, row["Name"]: total_consumption}).set_index(
                            "Speed (knots)")
    
                        compare_toggle = st.checkbox("Compare with another vessel", key=f"compare_toggle_{idx}",
                                                    disabled=not show_details)
                        if compare_toggle:
                            compare_vessel = st.selectbox("Select vessel to compare",
                                                        [v for i, v in enumerate(vessel_data['Name']) if i != idx],
                                                        key=f"compare_{idx}", disabled=not show_details)
                            compare_row = vessel_data[vessel_data['Name'] == compare_vessel].iloc[0]
                            compare_ref_consumption = compare_row["Main_Engine_Consumption_MT_per_day"] + compare_row[
                                "Generator_Consumption_MT_per_day"]
                            compare_total_consumption = [
                                compare_ref_consumption * (speed / base_speed) ** 3 for speed in
                                speed_range]
                            df_curve[compare_vessel] = compare_total_consumption
    
                        st.line_chart(df_curve)
    
                        vessel_data.at[idx, "Main_Engine_Consumption_MT_per_day"] = st.number_input(
                            "Main Engine (tons/day)", value=row["Main_Engine_Consumption_MT_per_day"], key=f"me_{idx}",
                            help="Daily fuel consumption of the main engine in metric tons.", disabled=not show_details)
                        vessel_data.at[idx, "Generator_Consumption_MT_per_day"] = st.number_input(
                            "Generator (tons/day)", value=row["Generator_Consumption_MT_per_day"], key=f"gen_{idx}",
                            help="Daily fuel consumption of onboard generators in metric tons.", disabled=not show_details)
                        c1, c2 = st.columns(2)
                        with c1:
                            vessel_data.at[idx, "Boil_Off_Rate_percent"] = st.number_input("Boil Off Rate (%)",
                                                                                        value=row["Boil_Off_Rate_percent"],
                                                                                        key=f"bor_{idx}",
                                                                                        help="Percentage of cargo volume lost due to boil-off.",
                                                                                        disabled=not show_details)
                        with c2:
                            vessel_data.at[idx, "CII_Rating"] = st.selectbox("CII Rating",
                                                                            options=["A", "B", "C", "D", "E"],
                                                                            index=["A", "B", "C", "D", "E"].index(
                                                                                row["CII_Rating"]),
                                                                            key=f"cii_{idx}",
                                                                            help="Carbon Intensity Indicator Rating.",
                                                                            disabled=not show_details)
                            vessel_data.at[idx, "FuelEU_GHG_Compliance"] = st.number_input(
                                "FuelEU GHG Intensity (gCO2e/MJ)", value=row["FuelEU_GHG_Compliance"], key=f"ghg_{idx}",
                                help="GHG intensity of the vessel according to FuelEU regulations.",
                                disabled=not show_details) 


    def page_2():
            st.title("Deployment Simulation")
            col_button_empty, col_button = st.columns([5, 1])
            with col_button:
                if st.button("Go Back to Main"):
                    st.session_state.page = "main"
            st.write("This is the content of Deployment Simulation.")
            vessel_data = pd.DataFrame({
    "Vessel_ID": range(1, 11),
    "Name": [f"LNG Carrier {chr(65 + i)}" for i in range(10)],
    "Length_m": [295] * 10,
    "Beam_m": [46] * 10,
    "Draft_m": [11.5] * 10,
    "Capacity_CBM": [160000] * 10,
    "FuelEU_GHG_Compliance": [65, 65, 65, 80, 80, 80, 95, 95, 95, 95],
    "CII_Rating": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
    "Main_Engine_Consumption_MT_per_day": [70, 72, 74, 85, 88, 90, 100, 102, 105, 107],
    "Generator_Consumption_MT_per_day": [5, 5, 5, 6, 6, 6, 7, 7, 7, 7],
    "Boil_Off_Rate_percent": [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.07, 0.07, 0.07, 0.07],
    "Margin": [2000] * 10
})
            # Inputs (replace with your actual input collection)
            carbon_calc_method = st.selectbox("Carbon Calculation Method", ["Fixed Rate", "Boil Off Rate"])
            ets_price = st.number_input("ETS Price (USD/MT CO2)", value=75)
            lng_bunker_price = st.number_input("LNG Bunker Price (USD/MT)", value=600)
            required_ghg_intensity = st.number_input("Required GHG Intensity", value=50)
            penalty_per_excess_unit = st.number_input("Penalty per Excess GHG Unit (USD)", value=1000)
            base_spot_rate = st.number_input("Base Spot Rate (USD/day)", value=120000)

            st.header("Deployment Simulation Results")

            total_co2_emissions = []
            results = []

            for index, vessel in vessel_data.iterrows():
                ref_total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]
                adjusted_fuel = ref_total_fuel
                if carbon_calc_method == "Boil Off Rate":
                    adjusted_fuel = vessel["Boil_Off_Rate_percent"] * vessel["Capacity_CBM"] / 1000

                auto_co2 = adjusted_fuel * 3.114
                carbon_cost = auto_co2 * ets_price
                fuel_cost = adjusted_fuel * lng_bunker_price
                margin_cost = vessel["Margin"]

                ghg_penalty = 0
                if vessel["FuelEU_GHG_Compliance"] > required_ghg_intensity:
                    excess = vessel["FuelEU_GHG_Compliance"] - required_ghg_intensity
                    ghg_penalty = excess * penalty_per_excess_unit

                breakeven = fuel_cost + carbon_cost + margin_cost + ghg_penalty

                results.append({
                    "Vessel": vessel["Name"],
                    "Fuel Cost ($/day)": f"{fuel_cost:,.1f}",
                    "Carbon Cost ($/day)": f"{carbon_cost:,.1f}",
                    "GHG Penalty ($/day)": f"{ghg_penalty:,.1f}",
                    "Margin ($/day)": f"{margin_cost:,.1f}",
                    "Breakeven Spot ($/day)": f"{breakeven:,.1f}",
                    "Decision": "✅ Spot Recommended" if base_spot_rate > breakeven else "❌ TC/Idle Preferred"
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

    def page_3():
            st.title("Voyage Simulation Advisor")
            col_button_empty, col_button = st.columns([5, 1])
            with col_button:
                if st.button("Go Back to Main"):
                    st.session_state.page = "main"
                
            vessel_data = pd.DataFrame({
    "Vessel_ID": range(1, 11),
    "Name": [f"LNG Carrier {chr(65 + i)}" for i in range(10)],
    "Length_m": [295] * 10,
    "Beam_m": [46] * 10,
    "Draft_m": [11.5] * 10,
    "Capacity_CBM": [160000] * 10,
    "FuelEU_GHG_Compliance": [65, 65, 65, 80, 80, 80, 95, 95, 95, 95],
    "CII_Rating": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
    "Main_Engine_Consumption_MT_per_day": [70, 72, 74, 85, 88, 90, 100, 102, 105, 107],
    "Generator_Consumption_MT_per_day": [5, 5, 5, 6, 6, 6, 7, 7, 7, 7],
    "Boil_Off_Rate_percent": [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.07, 0.07, 0.07, 0.07],
    "Margin": [2000] * 10
})
             # Input Prices
            p = st.number_input("Fuel Price (USD/MT)", value=700)
            ets_price = st.number_input("ETS Price (USD/MT of CO2 equivalent)", value=80)
            

            voyage_distance = st.number_input("Voyage Distance (nautical miles)", value=5000)
            freight_rate = st.number_input("Freight Rate (USD/day)", value=60000)

            for idx, vessel in vessel_data.iterrows():
                with st.expander(f"🛳️ {vessel['Name']} Voyage Simulation"):
                    speeds = np.arange(10, 20.5, 0.5)
                    sim_results = []
                    for speed in speeds:
                        voyage_days = voyage_distance / (speed * 24)
                        total_consumption = (vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]) * (speed / st.session_state.get('assumed_speed', 11.0)) ** 3 * voyage_days
                        fuel_cost = total_consumption * p
                        ets_cost = total_consumption * 3.114 * ets_price
                        total_cost = fuel_cost + ets_cost + vessel["Margin"] * voyage_days
                        tce = (freight_rate * voyage_days - total_cost) / voyage_days

                        sim_results.append({
                            "Speed (knots)": speed,
                            "Voyage Days": voyage_days,
                            "Fuel Consumption (MT)": total_consumption,
                            "Fuel Cost ($)": fuel_cost,
                            "ETS Cost ($)": ets_cost,
                            "Total Cost ($)": total_cost,
                            "TCE ($/day)": tce
                            })
                    sim_df = pd.DataFrame(sim_results)
                    best_speed_row = sim_df.loc[sim_df['TCE ($/day)'].idxmax()]
                    best_speed = best_speed_row["Speed (knots)"]

                    st.dataframe(sim_df.style.apply(lambda x: ["background-color: lightgreen" if x["Speed (knots)"] == best_speed else "" for _ in x], axis=1))
                    st.success(f"Optimal Economical Speed: {best_speed:.1f} knots with TCE of ${best_speed_row['TCE ($/day)']:.2f}/day")


    def page_4():
        st.title("Market Condition")
        col_button_empty, col_button = st.columns([5, 1])
        with col_button:
            if st.button("Go Back to Main"):
                st.session_state.page = "main"
        # Use sliders for the input fields and update session state
        st.session_state.scenario_name = st.text_input("Scenario Name", value=st.session_state.get('scenario_name', "My Scenario"))
        st.session_state.ets_price = st.slider("EU ETS Carbon Price (€/t CO₂)", 60, 150, st.session_state.get('ets_price', 95))
        st.session_state.lng_bunker_price = st.slider("LNG Bunker Price ($/ton)", 600, 1000, st.session_state.get('lng_bunker_price', 730))
        st.session_state.fleet_size_number_supply = st.slider("Fleet Size (# of Ships)", 1, 5000, st.session_state.get('fleet_size_number_supply', 3131), step=1)
        st.session_state.fleet_size_dwt_supply_in_dwt_million = st.slider("Supply (M DWT)", 100.0, 500.0, st.session_state.get('fleet_size_dwt_supply_in_dwt_million', 254.1), step=0.1)
        st.session_state.utilization_constant = st.slider("Utilization Factor", 0.0, 1.0, st.session_state.get('utilization_constant', 0.95), step=0.01)
        st.session_state.assumed_speed = st.slider("Speed (knots)", 5.0, 20.0, st.session_state.get('assumed_speed', 11.0), step=0.1)
        st.session_state.sea_margin = st.slider("Sea Margin (%)", 0.0, 0.1, st.session_state.get('sea_margin', 0.05), step=0.01)
        st.session_state.assumed_laden_days = st.slider("Laden Days Fraction", 0.0, 1.0, st.session_state.get('assumed_laden_days', 0.4), step=0.01)
        st.session_state.demand_billion_ton_mile = st.slider("Demand (Bn Ton Mile)", 1000.0, 20000.0, st.session_state.get('demand_billion_ton_mile', 10396.0), step=10.0)
        st.session_state.auto_tightness = st.checkbox("Auto-calculate market tightness", value=st.session_state.get('auto_tightness', True))
        st.session_state.base_spot_rate = st.slider("Spot Rate (USD/day)", 5000, 150000, st.session_state.get('base_spot_rate', 60000), step=1000)
        st.session_state.base_tc_rate = st.slider("TC Rate (USD/day)", 5000, 140000, st.session_state.get('base_tc_rate', 50000), step=1000)
        st.session_state.carbon_calc_method = st.radio("Carbon Cost Based On", ["Main Engine Consumption", "Boil Off Rate"], index=["Main Engine Consumption", "Boil Off Rate"].index(st.session_state.get('carbon_calc_method', 'Main Engine Consumption')))



        # Get values from session state or use defaults
        scenario_name = st.session_state.get("scenario_name", "My Scenario")
        ets_price = st.session_state.get("ets_price", 95)
        lng_bunker_price = st.session_state.get("lng_bunker_price", 730)
        fleet_size_number_supply = st.session_state.get("fleet_size_number_supply", 3131)
        fleet_size_dwt_supply_in_dwt_million = st.session_state.get("fleet_size_dwt_supply_in_dwt_million", 254.1)
        utilization_constant = st.session_state.get("utilization_constant", 0.95)
        assumed_speed = st.session_state.get("assumed_speed", 11.0)
        sea_margin = st.session_state.get("sea_margin", 0.05)
        assumed_laden_days = st.session_state.get("assumed_laden_days", 0.4)
        demand_billion_ton_mile = st.session_state.get("demand_billion_ton_mile", 10396.0)
        auto_tightness = st.session_state.get("auto_tightness", True)
        base_spot_rate = st.session_state.get("base_spot_rate", 60000)
        base_tc_rate = st.session_state.get("base_tc_rate", 50000)
        carbon_calc_method = st.session_state.get("carbon_calc_method", "Main Engine Consumption")
        d = st.session_state.get("carbon_calc_method", "Main Engine Consumption")

        # Tightness calculation (moved inside page_4)

        dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
        distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
        productive_laden_days_per_year = assumed_laden_days * 365
        maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
        equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile

        if auto_tightness:
            market_tightness = min(max(0.3 + (equilibrium / demand_billion_ton_mile), 0.0), 1.0)
        else:
            market_tightness = st.session_state.get("manual_tightness", 0.5)

        sensitivity = abs(equilibrium / demand_billion_ton_mile)

        # Display the market inputs and equilibrium calculations.
        st.header("Market Inputs")
        st.write(f"Scenario Name: {scenario_name}")
        st.write(f"EU ETS Carbon Price: {ets_price} €/t CO₂")
        st.write(f"LNG Bunker Price: {lng_bunker_price} $/ton")

        st.subheader("Freight Market Inputs")
        st.write(f"Fleet Size: {fleet_size_number_supply} Ships")
        st.write(f"Supply: {fleet_size_dwt_supply_in_dwt_million} M DWT")
        st.write(f"Utilization Factor: {utilization_constant}")
        st.write(f"Assumed Speed: {assumed_speed} knots")
        st.write(f"Sea Margin: {sea_margin * 100}%")
        st.write(f"Laden Days Fraction: {assumed_laden_days}")
        st.write(f"Demand: {demand_billion_ton_mile} Bn Ton Mile")
        st.write(f"Auto-calculate market tightness: {auto_tightness}")

        st.header("Equilibrium Calculations")
        st.write(f"DWT Utilization: {dwt_utilization:.1f} MT")
        st.write(f"Max Supply: {maximum_supply_billion_ton_mile:.1f} Bn Ton Mile")
        st.write(f"Equilibrium: {equilibrium:.1f} Bn Ton Mile")
        st.write(f"Market Condition: { 'Excess Supply' if equilibrium < 0 else 'Excess Demand'}")
        st.write(f"Market Tightness: {market_tightness:.2f}")
        st.write(f"Market Sensitivity: {sensitivity:.2%}")
        st.header("Base Rates")
        st.write(f"Spot Rate: {base_spot_rate} USD/day")
        st.write(f"TC Rate: {base_tc_rate} USD/day")
        st.write(f"Carbon Cost Based On: {carbon_calc_method}")









    def page_5_content(): # Renamed function to avoid conflict
        st.title("Feright Rate Monitoring")
        col_button_empty, col_button = st.columns([5, 1])
        with col_button:
                if st.button("Go Back to Main"):
                    st.session_state.page = "main"
        # Sidebar for navigation (This is what you wanted on Page 5)
        page_5_sidebar = st.sidebar.radio("Select Data", ["Equilibrium Calculator", "Vessel Profile Data", "LNG Market Trends", "Yearly Simulation Data"])

        if page_5_sidebar == "Equilibrium Calculator":
            st.subheader("Shipping Market Equilibrium Calculator")

            # Inputs
            fleet_size_number_supply = st.number_input("Fleet Size (Number of Ships)", value=3131, step=1, format="%d")
            fleet_size_dwt_supply_in_dwt_million = st.number_input("Fleet Size Supply (Million DWT)", value=254.1, step=0.1)
            utilization_constant = st.number_input("Utilization Constant", value=0.95, step=0.01)

            assumed_speed = st.number_input("Assumed Speed (knots)", value=11.0, step=0.1)
            sea_margin = st.number_input("Sea Margin", value=0.05, step=0.01)

            assumed_laden_days = st.number_input("Assumed Laden Days Fraction", value=0.4, step=0.01)

            demand_billion_ton_mile = st.number_input("Demand (Billion Ton Mile)", value=10396.0, step=10.0)

            # Calculations
            dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
            distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
            productive_laden_days_per_year = assumed_laden_days * 365

            # Maximum Supply Calculation
            maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000

            # Equilibrium
            equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile
            result = "Excess Supply" if equilibrium < 0 else "Excess Demand"

            # Display results
            st.subheader("Results:")
            st.metric(label="DWT Utilization (tons)", value=f"{dwt_utilization:,.2f}")
            st.metric(label="Distance Travelled per Day (nm)", value=f"{distance_travelled_per_day:,.2f}")
            st.metric(label="Productive Laden Days per Year", value=f"{productive_laden_days_per_year:,.2f}")
            st.metric(label="Maximum Supply (Billion Ton Mile)", value=f"{maximum_supply_billion_ton_mile:,.2f}")
            st.metric(label="Equilibrium (Billion Ton Mile)", value=f"{equilibrium:,.2f}")
            st.metric(label="Market Condition", value=result)

            # Visualization
            fig, ax = plt.subplots()
            ax.bar(["Demand", "Supply"], [demand_billion_ton_mile, maximum_supply_billion_ton_mile], color=['blue', 'orange'])
            ax.set_ylabel("Billion Ton Mile")
            ax.set_title("Supply vs Demand")
            st.pyplot(fig)

            # col_empty, col_back_button = st.columns([5, 1])
            # with col_back_button:
            #     if st.button("Go Back to Main"):
            #         st.session_state.page = "main"

        elif page_5_sidebar == "Vessel Profile Data":
            st.title("🚢 Vessel Profile Data")

            # Vessel Data
            vessel_data = pd.DataFrame({
                "Vessel_ID": range(1, 11),
                "Name": [
                    "LNG Carrier Alpha", "LNG Carrier Beta", "LNG Carrier Gamma", "LNG Carrier Delta",
                    "LNG Carrier Epsilon", "LNG Carrier Zeta", "LNG Carrier Theta", "LNG Carrier Iota",
                    "LNG Carrier Kappa", "LNG Carrier Lambda"
                ],
                "Sister_Ship_Group": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
                "Capacity_CBM": [160000] * 10,
                "FuelEU_GHG_Compliance": [65, 65, 65, 80, 80, 80, 95, 95, 95, 95],
                "CII_Rating": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
                "Fuel_Consumption_MT_per_day": [70, 72, 74, 85, 88, 90, 100, 102, 105, 107]
            })

            # Input for Fuel Price and Voyage Days
            fuel_price = st.number_input("Enter Fuel Price (per MT in USD)", min_value=0.0, value=500.0, step=10.0)
            voyage_days = int(st.number_input("Enter Voyage Days", min_value=1, value=10, step=1))
            freight_rate_per_day = float(st.number_input("Enter Freight Rate per Day (USD)", min_value=0.0, value=100000.0, step=1000.0))

            # Calculate Fuel Cost Per Day and Total Cost
            vessel_data["Fuel_Cost_per_Day"] = (vessel_data["Fuel_Consumption_MT_per_day"] * fuel_price).astype(int)
            vessel_data["Total_Voyage_Cost"] = (vessel_data["Fuel_Cost_per_Day"] * voyage_days).astype(int)

            # Calculate Freight Earnings and Profit
            vessel_data["Total_Freight_Earnings"] = freight_rate_per_day * voyage_days
            vessel_data["Total_Profit"] = vessel_data["Total_Freight_Earnings"] - vessel_data["Total_Voyage_Cost"]

            # Ensure all values are numeric and format correctly
            numeric_columns = ["Capacity_CBM", "FuelEU_GHG_Compliance", "Fuel_Consumption_MT_per_day", "Fuel_Cost_per_Day", "Total_Voyage_Cost", "Total_Freight_Earnings", "Total_Profit"]
            vessel_data[numeric_columns] = vessel_data[numeric_columns].apply(pd.to_numeric)

            # Format table to display values in a single line and center-align
            st.markdown(
                vessel_data.style.set_properties(
                    **{'text-align': 'center', 'white-space': 'nowrap'}
                ).set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center')]}
                ]).format({col: "{:,.0f}" for col in numeric_columns}).to_html(),
                unsafe_allow_html=True
            )

            # Show a summary of total fleet fuel cost
            total_fuel_cost = vessel_data["Fuel_Cost_per_Day"].sum()
            total_voyage_cost = vessel_data["Total_Voyage_Cost"].sum()
            total_freight_earnings = int(freight_rate_per_day * voyage_days)
            total_profit = total_freight_earnings - total_voyage_cost

            # st.metric(label="Total Fleet Fuel Cost per Day (USD)", value=f"<span class='math-inline'>${total_fuel_cost:,}</span>")
            # st.metric(label="Total Voyage Cost (USD)", value=f"<span class='math-inline'>${total_voyage_cost:,}</span>")
            # st.metric(label="Total Freight Earnings (USD)", value=f"<span class='math-inline'>${total_freight_earnings:,}</span>")
            # st.metric(label="Total Profit (USD)", value=f"<span class='math-inline'>${total_profit:,}</span>")

            st.metric(label="Total Fleet Fuel Cost per Day (USD)", value=f"${total_fuel_cost:,.0f}")
            st.metric(label="Total Voyage Cost (USD)", value=f"${total_voyage_cost:,.0f}")
            st.metric(label="Total Freight Earnings (USD)", value=f"${total_freight_earnings:,.0f}")
            st.metric(label="Total Profit (USD)", value=f"${total_profit:,.0f}")
            col_empty, col_back_button = st.columns([5, 1])
            with col_back_button:
                if st.button("Go Back to Main"):
                    st.session_state.page = "main"

        elif page_5_sidebar == "LNG Market Trends":
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
                    st.error("⚠️ 'Date' column not found in the dataset.")
        
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
                            showlegend=True,  # Set showlegend to True at the layout level
                            height=300 * num_plots,
                        )
        
                        st.plotly_chart(fig, use_container_width=True)
        
                    else:
                        st.warning("Please select at least one data column.")
        
        
        
                    col_empty, col_back_button = st.columns([5, 1])
                    with col_back_button:
                        if st.button("Go Back to Main"):
                            st.session_state.page = "main"
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                        
        
        elif page_5_sidebar == "Yearly Simulation Data":
                st.title("📊 Yearly Simulation Data")
            
                base_url = "https://docs.google.com/spreadsheets/d/1kySjcfv1jMkDRrqAD9qS10KjIs5H1Vdu/gviz/tq?tqx=out:csv&sheet=Yearly%20equilibrium"
                
                try:
                    df_yearly_sim = pd.read_csv(base_url, dtype=str)
            
                    if "Year" in df_yearly_sim.columns:
                        df_yearly_sim["Year"] = pd.to_datetime(df_yearly_sim["Year"], format="%Y", errors='coerce').dt.year
                        df_yearly_sim = df_yearly_sim.dropna(subset=["Year"]).sort_values(by="Year")
                    else:
                        st.error("⚠️ 'Year' column not found in the dataset.")
            
                    for col in df_yearly_sim.columns:
                        if col != "Year":
                            df_yearly_sim[col] = pd.to_numeric(df_yearly_sim[col], errors='coerce').fillna(0)
            
                    available_columns = [col for col in df_yearly_sim.columns if col != "Year"]
                    variable_option = st.multiselect("Select Data Columns", available_columns, default=available_columns[:2] if available_columns else [])
            
                    start_year = st.number_input("Select Start Year", int(df_yearly_sim["Year"].min()), int(df_yearly_sim["Year"].max()), int(df_yearly_sim["Year"].min()))
                    end_year = st.number_input("Select End Year", int(df_yearly_sim["Year"].min()), int(df_yearly_sim["Year"].max()), int(df_yearly_sim["Year"].max()))
                    df_filtered = df_yearly_sim[(df_yearly_sim["Year"] >= start_year) & (df_yearly_sim["Year"] <= end_year)]
            
                    # User selects a specific year
                    selected_year = st.number_input("Select a Year to Display Data", int(df_yearly_sim["Year"].min()), int(df_yearly_sim["Year"].max()), int(df_yearly_sim["Year"].max()))
            
                    if variable_option:
                        selected_data = df_yearly_sim[df_yearly_sim["Year"] == selected_year][variable_option]
                        st.subheader(f"Data for {selected_year}")
                        st.write(selected_data)
            
                    if len(variable_option) > 0 and not df_filtered.empty:
                        num_plots = (len(variable_option) + 1) // 2
                        specs = [[{'secondary_y': True}] for _ in range(num_plots)]
                        fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.3, specs=specs)
            
                        for i in range(0, len(variable_option), 2):
                            row_num = (i // 2) + 1
            
                            # Plot the first column in the pair
                            fig.add_trace(
                                go.Scatter(
                                    x=df_filtered["Year"],
                                    y=df_filtered[variable_option[i]],
                                    mode='lines',
                                    name=variable_option[i],
                                    hovertemplate='Year: %{x}<br>Value: %{y}<extra></extra>',
                                    showlegend=True,
                                    legendgroup=variable_option[i],
                                ),
                                row=row_num,
                                col=1,
                                secondary_y=False,
                            )
                            fig.update_yaxes(title_text=variable_option[i], row=row_num, col=1, secondary_y=False)
            
                            # Plot the second column in the pair (if it exists)
                            if i + 1 < len(variable_option):
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_filtered["Year"],
                                        y=df_filtered[variable_option[i + 1]],
                                        mode='lines',
                                        name=variable_option[i + 1],
                                        hovertemplate='Year: %{x}<br>Value: %{y}<extra></extra>',
                                        showlegend=True,
                                        legendgroup=variable_option[i + 1],
                                    ),
                                    row=row_num,
                                    col=1,
                                    secondary_y=True,
                                )
                                fig.update_yaxes(title_text=variable_option[i + 1], row=row_num, col=1, secondary_y=True)
            
                        fig.update_layout(
                            title="Yearly Simulation Trends",
                            xaxis=dict(
                                title=f"Year Range: {start_year} to {end_year}",
                                tickangle=45,
                                tickformatstops=[dict(dtickrange=[None, None], value="%Y")],
                                range=[df_filtered["Year"].min(), df_filtered["Year"].max()]
                            ),
                            hovermode="x unified",
                            showlegend=True,
                            height=300 * num_plots,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(variable_option) == 0:
                        st.warning("Please select at least one data column to display the plot.")
                    elif df_filtered.empty:
                        st.warning("No data available for the selected year range.")
            
                    col_empty, col_back_button = st.columns([5, 1])
                    with col_back_button:
                        if st.button("Go Back to Main"):
                            st.session_state.page = "main"
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