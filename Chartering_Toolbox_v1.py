import streamlit as st
import numpy as np
import random
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
        st.header("🛠️ Vessel Input Section")

        # --- Performance Profiles ---
        performance_profiles = {
            "good": {"a": 10, "b": 2, "c": 0.05, "d": 0.001},
            "medium": {"a": 12, "b": 2.5, "c": 0.06, "d": 0.0012},
            "poor": {"a": 15, "b": 3, "c": 0.07, "d": 0.0015},
        }

        # --- Helper Functions ---
        def calculate_fuel_consumption(speed, profile="good"):
            """Calculates fuel consumption based on speed and performance profile."""
            coeffs = performance_profiles[profile]
            return coeffs["a"] + coeffs["b"] * speed + coeffs["c"] * speed**2 + coeffs["d"] * speed**3


        def update_fuel_consumption(edited_data, index):
            """Callback function to update fuel consumption based on speed changes."""
            if edited_data is not None: # Check if edited_data exists
                if "vessel_data" in st.session_state:
                    vessel_data = st.session_state["vessel_data"].copy()
                    if index < len(vessel_data):
                        if edited_data.get("data"):  # Use .get() for safety
                            edited_df = pd.DataFrame(edited_data["data"])
                            if "Speed (knots)" in edited_df.columns and "Fuel Consumption (tons/day)" in edited_df.columns:
                                edited_row = edited_df.iloc[0] # Assuming single row edit at a time
                                speed = edited_row["Speed (knots)"]
                                profile = vessel_data.at[index, "Performance_Profile"] # Get the profile for the current vessel
                                fuel_consumption = calculate_fuel_consumption(speed, profile)

                                # Update the 'Fuel Consumption' in the displayed editor data
                                current_editor_data = st.session_state.get(f"editor_{index}_data", pd.DataFrame())
                                if not current_editor_data.empty and len(edited_df) > 0:
                                    current_editor_data.loc[current_editor_data.index[0], "Fuel Consumption (tons/day)"] = fuel_consumption
                                    st.session_state[f"editor_{index}_data"] = current_editor_data

                                # Update the main vessel_data DataFrame
                                speed_col_name = f"Speed_{int(speed)}"
                                vessel_data.at[index, speed_col_name] = fuel_consumption
                                st.session_state["vessel_data"] = vessel_data
                    else:
                        st.warning(f"Vessel index {index} out of bounds.")


        # Ensure vessel_data exists in session state or create it
        if "vessel_data" not in st.session_state:
            st.session_state["vessel_data"] = pd.DataFrame({
                "Vessel_ID": range(1, 11),
                "Name": [f"LNG Carrier {chr(65 + i)}" for i in range(10)],
                "Length_m": [295] * 10,
                "Beam_m": [46] * 10,
                "Draft_m": [11.5] * 10,
                "Capacity_CBM": [160000] * 10,
                "FuelEU_GHG_Compliance": [65, 65, 65, 80, 80, 80, 95, 95, 95, 95],
                "CII_Rating": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
                "Boil_Off_Rate_percent": [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.07, 0.07, 0.07, 0.07],
                "Margin": [2000] * 10,
                "Performance_Profile": ["good"] * 10  # Initialize a default profile
            })

        vessel_data = st.session_state["vessel_data"]
        speed_range_default = list(range(8, 22))  # Default speed range for initial data

        col_button_empty, col_button = st.columns([5, 1])
        with col_button:
            if st.button("Go Back to Main", key="back_to_main_page1"):
                st.session_state.page = "main"

        cols = st.columns(2)
        for idx, row in vessel_data.iterrows():
            with cols[idx % 2].expander(f"🚢 {row['Name']}"):
                vessel_data.at[idx, "Name"] = st.text_input("Vessel Name", value=row["Name"], key=f"name_{idx}")
                vessel_data.at[idx, "Length_m"] = st.number_input("Length (m)", value=row["Length_m"], key=f"len_{idx}")
                vessel_data.at[idx, "Beam_m"] = st.number_input("Beam (m)", value=row["Beam_m"], key=f"beam_{idx}")
                vessel_data.at[idx, "Draft_m"] = st.number_input("Draft (m)", value=row["Draft_m"], key=f"draft_{idx}")
                vessel_data.at[idx, "Margin"] = st.number_input("Margin (USD/day)", value=row["Margin"], key=f"margin_{idx}")

                show_details = st.toggle("Show Performance Details", key=f"toggle_{idx}")
                if show_details:
                    st.subheader("✏️ Speed vs. Fuel Consumption (tons/day)")

                    # Get existing speed/consumption data or initialize with default
                    speed_cols = [col for col in vessel_data.columns if col.startswith("Speed_")]
                    if speed_cols:
                        speed_data = pd.DataFrame({
                            "Speed (knots)": [int(col.split("_")[1]) for col in speed_cols],
                            "Fuel Consumption (tons/day)": [row[col] for col in speed_cols]
                        }).sort_values(by="Speed (knots)").reset_index(drop=True)
                    else:
                        # Use default calculation for initial data
                        profile = vessel_data.at[idx, "Performance_Profile"]
                        coeffs = performance_profiles[profile]
                        profile_peaks = {"good": 125.0, "medium": 140.0, "poor": 155.0}
                        target_max = profile_peaks[profile]
                        raw_curve = [coeffs["a"] + coeffs["b"] * s + coeffs["c"] * s**2 + coeffs["d"] * s**3 for s in speed_range_default]
                        current_at_21 = raw_curve[-1]
                        scaling_factor = target_max / current_at_21
                        variation = np.random.uniform(0.97, 1.03, size=len(raw_curve))
                        scaled_curve = [val * scaling_factor * v for val, v in zip(raw_curve, variation)]
                        speed_data = pd.DataFrame({"Speed (knots)": speed_range_default, "Fuel Consumption (tons/day)": scaled_curve})

                    # Store and retrieve the editor data in session state
                    editor_key = f"editor_{idx}_data"
                    if editor_key not in st.session_state:
                        st.session_state[editor_key] = speed_data.copy()

                    edited_data = st.data_editor(
                        st.session_state[editor_key],
                        key=f"editor_{idx}",
                        num_rows="dynamic",
                        on_change=lambda: update_fuel_consumption(st.session_state.get(f"editor_{idx}_data"), idx)
                    )
                    st.session_state[editor_key] = edited_data  # Update session state with the edited data

                    try:
                        speeds = edited_data["Speed (knots)"].values if "Speed (knots)" in edited_data else []
                        consumptions = edited_data["Fuel Consumption (tons/day)"].values if "Fuel Consumption (tons/day)" in edited_data else []

                        if len(speeds) >= 2:  # Need at least two points to fit a curve
                            poly_coeffs = np.polyfit(speeds, consumptions, deg=3)
                            a, b, c, d = poly_coeffs[3], poly_coeffs[2], poly_coeffs[1], poly_coeffs[0]

                            st.markdown("### 📈 Fitted Cubic Curve Coefficients:")
                            st.markdown(f"**a** = {a:.3f} &nbsp;&nbsp; **b** = {b:.3f} &nbsp;&nbsp; **c** = {c:.3f} &nbsp;&nbsp; **d** = {d:.5f}")

                            smooth_speeds = np.linspace(min(speeds), max(speeds), 100)
                            fitted_curve = a + b * smooth_speeds + c * smooth_speeds**2 + d * smooth_speeds**3

                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=speeds,
                                y=consumptions,
                                mode='markers',
                                name='User Input',
                                marker=dict(size=8, color='blue')
                            ))

                            fig.add_trace(go.Scatter(
                                x=smooth_speeds,
                                y=fitted_curve,
                                mode='lines',
                                name='Fitted Curve',
                                line=dict(color='green', width=2)
                            ))

                            # === Optional: compare with other vessel's fitted curve ===
                            compare_toggle = st.checkbox("Compare with another vessel", key=f"compare_toggle_{idx}")
                            if compare_toggle:
                                compare_vessel = st.selectbox("Select vessel to compare", [v for i, v in enumerate(vessel_data['Name']) if i != idx], key=f"compare_{idx}")
                                compare_row = vessel_data[vessel_data["Name"] == compare_vessel].iloc[0]
                                compare_speeds_comp = [int(col.split("_")[1]) for col in compare_row.index if col.startswith("Speed_")]
                                compare_consumptions_comp = [compare_row[f"Speed_{s}"] for s in compare_speeds_comp]

                                if len(compare_speeds_comp) >= 2:
                                    try:
                                        comp_coeffs = np.polyfit(compare_speeds_comp, compare_consumptions_comp, deg=3)
                                        a2, b2, c2, d2 = comp_coeffs[3], comp_coeffs[2], comp_coeffs[1], comp_coeffs[0]
                                        smooth_speeds_comp = np.linspace(min(compare_speeds_comp), max(compare_speeds_comp), 100)
                                        compare_fitted = a2 + b2 * smooth_speeds_comp + c2 * smooth_speeds_comp**2 + d2 * smooth_speeds_comp**3

                                        fig.add_trace(go.Scatter(
                                            x=smooth_speeds_comp,
                                            y=compare_fitted,
                                            mode='lines',
                                            name=f"{compare_vessel} (Fitted)",
                                            line=dict(color='red', dash='dot')
                                        ))
                                    except Exception as e:
                                        st.warning(f"Could not fit comparison vessel: {e}")
                                else:
                                    st.warning(f"Not enough speed/consumption data for {compare_vessel} to compare.")

                            fig.update_layout(
                                title="Speed vs. Fuel Consumption",
                                xaxis_title="Speed (knots)",
                                yaxis_title="Fuel Consumption (tons/day)",
                                legend=dict(x=0.01, y=0.99)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Please input at least two speed/consumption data points to fit the curve.")

                    except Exception as e:
                        st.warning(f"Could not fit curve: {e}")

                    # Technical & regulatory inputs
                    c1, c2 = st.columns(2)
                    with c1:
                        vessel_data.at[idx, "Boil_Off_Rate_percent"] = st.number_input("Boil Off Rate (%)", value=row["Boil_Off_Rate_percent"], key=f"bor_{idx}")
                    with c2:
                        vessel_data.at[idx, "CII_Rating"] = st.selectbox("CII Rating", ["A", "B", "C", "D", "E"],
                                                                        index=["A", "B", "C", "D", "E"].index(row["CII_Rating"]),
                                                                        key=f"cii_{idx}")
                        vessel_data.at[idx, "FuelEU_GHG_Compliance"] = st.number_input("FuelEU GHG Intensity (gCO₂e/MJ)",
                                                                                        value=row["FuelEU_GHG_Compliance"],
                                                                                        key=f"ghg_{idx}",
                                                                                        help="GHG intensity of the vessel according to FuelEU regulations.")

        # Update session state with the modified vessel data
        st.session_state["vessel_data"] = vessel_data
    def page_2():
        st.title("Deployment Simulation")
    
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
    
        # col_input, col_results = st.columns([1, 2]) # Adjusted column widths
        col_input, col_results = st.columns([0.7, 2.3]) # Further adjusted column widths

    
        with col_input:
            st.subheader("Simulation Inputs")
            carbon_calc_method = st.selectbox("Carbon Calculation Method", ["Fixed Rate", "Boil Off Rate"])
            ets_price = st.number_input("ETS Price (USD/MT CO2)", value=75)
            lng_bunker_price = st.number_input("LNG Bunker Price (USD/MT)", value=600)
            required_ghg_intensity = st.number_input("Required GHG Intensity", value=50)
            penalty_per_excess_unit = st.number_input("Penalty per Excess GHG Unit (USD)", value=1000)
            base_spot_rate = st.number_input("Base Spot Rate (USD/day)", value=120000)
    
        with col_results:
            st.subheader("Deployment Simulation Results")
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
            st.dataframe(results_df, use_container_width=True) # Ensure dataframe uses available width

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
                        "Voyage Days": f"{voyage_days:.1f}",
                        "Fuel Consumption (MT)": total_consumption,
                        "Fuel Cost ($)": fuel_cost,
                        "ETS Cost ($)": ets_cost,
                        "Total Cost ($)": total_cost,
                        "TCE ($/day)": tce
                    })
                sim_df = pd.DataFrame(sim_results)
                best_speed_row = sim_df.loc[sim_df['TCE ($/day)'].idxmax()]
                best_speed = best_speed_row["Speed (knots)"]
    


                def highlight_best_speed(row):
                    # if row.name != sim_df.index.name:  # Check if it's not the header row
                    #     if float(row["Speed (knots)"]) == best_speed:
                    #         return ["background-color: lightgreen"] * len(row)
                    return [""] * len(row)

                st.dataframe(sim_df.style.apply(highlight_best_speed, axis=1).format(precision=1))
    
                # st.dataframe(sim_df.style.apply(highlight_best_speed, axis=1))
                st.success(f"Optimal Economical Speed: {best_speed:.1f} knots with TCE of ${best_speed_row['TCE ($/day)']:.2f}/day")
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
            st.session_state.scenario_name = st.text_input("Scenario Name", value=st.session_state.get('scenario_name', "My Scenario"), key="scenario_name_input")
            st.session_state.ets_price = st.slider("EU ETS Carbon Price (€/t CO₂)", 60, 150, st.session_state.get('ets_price', 95), key="ets_price_slider")
            st.session_state.lng_bunker_price = st.slider("LNG Bunker Price ($/ton)", 600, 1000, st.session_state.get('lng_bunker_price', 730), key="lng_bunker_price_slider")

            st.subheader("Freight Market Inputs")
            st.session_state.fleet_size_number_supply = st.slider("Fleet Size (# of Ships)", 1, 5000, st.session_state.get('fleet_size_number_supply', 3131), step=1, key="fleet_size_number_supply_slider")
            st.session_state.fleet_size_dwt_supply_in_dwt_million = st.slider("Supply (M DWT)", 100.0, 500.0, st.session_state.get('fleet_size_dwt_supply_in_dwt_million', 254.1), step=0.1, key="fleet_size_dwt_supply_in_dwt_million_slider")
            st.session_state.utilization_constant = st.slider("Utilization Factor", 0.0, 1.0, st.session_state.get('utilization_constant', 0.95), step=0.01, key="utilization_constant_slider")
            st.session_state.assumed_speed = st.slider("Speed (knots)", 5.0, 20.0, st.session_state.get('assumed_speed', 11.0), step=0.1, key="assumed_speed_slider")
            st.session_state.sea_margin = st.slider("Sea Margin (%)", 0.0, 0.1, st.session_state.get('sea_margin', 0.05), step=0.01, key="sea_margin_slider")
            st.session_state.assumed_laden_days = st.slider("Laden Days Fraction", 0.0, 1.0, st.session_state.get('assumed_laden_days', 0.4), step=0.01, key="assumed_laden_days_slider")
            st.session_state.demand_billion_ton_mile = st.slider("Demand (Bn Ton Mile)", 1000.0, 20000.0, st.session_state.get('demand_billion_ton_mile', 10396.0), step=10.0, key="demand_billion_ton_mile_slider")
            st.session_state.auto_tightness = st.checkbox("Auto-calculate market tightness", value=st.session_state.get('auto_tightness', True), key="auto_tightness_checkbox")
            if not st.session_state.auto_tightness:
                st.session_state.manual_tightness = st.slider("Manual Market Tightness", 0.0, 1.0, st.session_state.get("manual_tightness", 0.5), step=0.01, key="manual_tightness_slider")
            st.session_state.base_spot_rate = st.slider("Spot Rate (USD/day)", 5000, 150000, st.session_state.get('base_spot_rate', 60000), step=1000, key="base_spot_rate_slider")
            st.session_state.base_tc_rate = st.slider("TC Rate (USD/day)", 5000, 140000, st.session_state.get('base_tc_rate', 50000), step=1000, key="base_tc_rate_slider")
            st.session_state.carbon_calc_method = st.radio("Carbon Cost Based On", ["Main Engine Consumption", "Boil Off Rate"], index=["Main Engine Consumption", "Boil Off Rate"].index(st.session_state.get('carbon_calc_method', 'Main Engine Consumption')), key="carbon_calc_method_radio")

        with col_results:
            st.header("Market Inputs")
            st.write(f"Scenario Name: {st.session_state.get('scenario_name', 'My Scenario')}")
            st.write(f"EU ETS Carbon Price: {st.session_state.get('ets_price', 95)} €/t CO₂")
            st.write(f"LNG Bunker Price: {st.session_state.get('lng_bunker_price', 730)} $/ton")
            st.subheader("Freight Market Inputs")
            st.write(f"Fleet Size: {st.session_state.get('fleet_size_number_supply', 3131)} Ships")
            st.write(f"Supply: {st.session_state.get('fleet_size_dwt_supply_in_dwt_million', 254.1)} M DWT")
            st.write(f"Utilization Factor: {st.session_state.get('utilization_constant', 0.95)}")
            st.write(f"Assumed Speed: {st.session_state.get('assumed_speed', 11.0)} knots")
            st.write(f"Sea Margin: {st.session_state.get('sea_margin', 0.05) * 100}%")
            st.write(f"Laden Days Fraction: {st.session_state.get('assumed_laden_days', 0.4)}")
            st.write(f"Demand: {st.session_state.get('demand_billion_ton_mile', 10396.0)} Bn Ton Mile")
            st.write(f"Auto-calculate market tightness: {st.session_state.get('auto_tightness', True)}")
            st.subheader("Equilibrium Calculations")
            dwt_utilization = (st.session_state.fleet_size_dwt_supply_in_dwt_million * 1_000_000 / st.session_state.fleet_size_number_supply) * st.session_state.utilization_constant
            distance_travelled_per_day = st.session_state.assumed_speed * 24 * (1 - st.session_state.sea_margin)
            productive_laden_days_per_year = st.session_state.assumed_laden_days * 365
            maximum_supply_billion_ton_mile = st.session_state.fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
            equilibrium = st.session_state.demand_billion_ton_mile - maximum_supply_billion_ton_mile

            if st.session_state.auto_tightness:
                market_tightness = min(max(0.3 + (equilibrium / st.session_state.demand_billion_ton_mile), 0.0), 1.0)
            else:
                market_tightness = st.session_state.get("manual_tightness", 0.5)

            sensitivity = abs(equilibrium / st.session_state.demand_billion_ton_mile)

            st.write(f"DWT Utilization: {dwt_utilization:.1f} MT")
            st.write(f"Max Supply: {maximum_supply_billion_ton_mile:.1f} Bn Ton Mile")
            st.write(f"Equilibrium: {equilibrium:.1f} Bn Ton Mile")
            st.write(f"Market Condition: { 'Excess Supply' if equilibrium < 0 else 'Excess Demand'}")
            st.write(f"Market Tightness: {market_tightness:.2f}")
            st.write(f"Market Sensitivity: {sensitivity:.2%}")

            st.subheader("Base Rates")
            st.write(f"Spot Rate: {st.session_state.base_spot_rate} USD/day")
            st.write(f"TC Rate: {st.session_state.base_tc_rate} USD/day")
            st.write(f"Carbon Cost Based On: {st.session_state.carbon_calc_method}")




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
