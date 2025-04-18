import streamlit as st
import numpy as np
import pandas as pd
import json

st.set_page_config(page_title="LNG 10-Vessel Deployment Tool", layout="wide")

if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = {}

def get_value(key, default):
    return st.session_state.loaded_data.get(key, default)

# ----------------------- SIDEBAR -----------------------
st.sidebar.title("⚙️Market Inputs and Equilibrium calculator")
scenario_name = st.sidebar.text_input("Scenario Name", value=get_value("scenario_name", "My Scenario"))
ets_price = st.sidebar.slider("EU ETS Carbon Price (€/t CO₂)", 60, 150, get_value("ets_price", 95))
lng_bunker_price = st.sidebar.slider("LNG Bunker Price ($/ton)", 600, 1000, get_value("lng_bunker_price", 730))

st.sidebar.header("💡 Freight Market Inputs")
fleet_size_number_supply = st.sidebar.number_input("Fleet Size (# of Ships)", value=get_value("fleet_size_number_supply", 3131), step=1, format="%d")
fleet_size_dwt_supply_in_dwt_million = st.sidebar.number_input("Supply (M DWT)", value=get_value("fleet_size_dwt_supply_in_dwt_million", 254.1), step=0.1)
utilization_constant = st.sidebar.number_input("Utilization Factor", value=get_value("utilization_constant", 0.95), step=0.01)
assumed_speed = st.sidebar.number_input("Speed (knots)", value=get_value("assumed_speed", 11.0), step=0.1)
sea_margin = st.sidebar.number_input("Sea Margin (%)", value=get_value("sea_margin", 0.05), step=0.01)
assumed_laden_days = st.sidebar.number_input("Laden Days Fraction", value=get_value("assumed_laden_days", 0.4), step=0.01)
demand_billion_ton_mile = st.sidebar.number_input("Demand (Bn Ton Mile)", value=get_value("demand_billion_ton_mile", 10396.0), step=10.0)
auto_tightness = st.sidebar.checkbox("Auto-calculate market tightness", value=get_value("auto_tightness", True))

# Tightness calculation
dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
productive_laden_days_per_year = assumed_laden_days * 365
maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile

if auto_tightness:
    market_tightness = min(max(0.3 + (equilibrium / demand_billion_ton_mile), 0.0), 1.0)
else:
    market_tightness = st.sidebar.slider("Manual Tightness (0-1)", 0.0, 1.0, get_value("manual_tightness", 0.5))

with st.sidebar.expander("🔍 Market Calculations"):
    st.markdown(f"**DWT Utilization:** {dwt_utilization:,.1f} MT")
    st.markdown(f"**Max Supply:** {maximum_supply_billion_ton_mile:,.1f} Bn Ton Mile")
    st.markdown(f"**Equilibrium:** {equilibrium:,.1f} Bn Ton Mile")
    market_status = 'Excess Supply' if equilibrium < 0 else 'Excess Demand'
    status_color = 'red' if market_status == 'Excess Supply' else 'green'
    st.markdown(f"**Market Condition:** <span style='color:{status_color}'>{market_status}</span>", unsafe_allow_html=True)
    st.markdown(f"**Tightness:** {market_tightness:.2f}")

sensitivity = abs(equilibrium / demand_billion_ton_mile)
with st.sidebar.expander("📊 Market Sensitivity", expanded=True):
    st.metric(label="Market Sensitivity", value=f"{sensitivity:.2%}")
    st.caption("**Market Sensitivity**: Measures how tight or loose the market is relative to total demand.")

base_spot_rate = st.sidebar.slider("Spot Rate (USD/day)", 5000, 150000, get_value("base_spot_rate", 60000), step=1000)
base_tc_rate = st.sidebar.slider("TC Rate (USD/day)", 5000, 140000, get_value("base_tc_rate", 50000), step=1000)
carbon_calc_method = st.sidebar.radio("Carbon Cost Based On", ["Main Engine Consumption", "Boil Off Rate"], index=["Main Engine Consumption", "Boil Off Rate"].index(get_value("carbon_calc_method", "Main Engine Consumption")))

# ----------------------- MAIN PANEL -----------------------
st.title("LNG Fleet Deployment Simulator")

# Evans & Marlow Speed Calculator Section
st.header("🔍 Evans & Marlow Optimal Speed Calculator")
with st.expander("Calculate optimal speed based on freight rate, fuel price, and distance"):
    R = st.number_input("Freight Rate (USD/ton-mile)", min_value=0.0, value=50.0)
    p = st.number_input("Fuel Price (USD/ton)", min_value=0.0, value=730.0)
    k = st.number_input("Ship Fuel Constant (k)", min_value=0.0, value=0.0002)
    d = st.number_input("Voyage Distance (nm)", min_value=0.0, value=5000.0)

    if all([R > 0, p > 0, k > 0, d > 0]):
        S_opt = ((R / (3 * p * k * d)) ** 0.5)
        st.success(f"Recommended Optimal Speed: {S_opt:.1f} knots")
        if st.button("Apply this speed to Freight Market Inputs"):
            st.session_state['assumed_speed'] = S_opt
    else:
        st.info("Please enter positive values for all fields.")



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

# ----------------------- Vessel Input Section -----------------------

import numpy as np
import random
import pandas as pd
import plotly.graph_objects as go

st.header("🛠️ Vessel Input Section")
speed_range = list(range(8, 22))  # Speeds from 8 to 21 knots

# Performance profiles
performance_profiles = {
    "good":   {"a": 20.0, "b": -1.0, "c": 0.5, "d": 0.010},
    "medium": {"a": 30.0, "b": -0.5, "c": 0.8, "d": 0.015},
    "poor":   {"a": 40.0, "b":  0.0, "c": 1.2, "d": 0.020},
}

# Assign unique profile per vessel if not already assigned
if "Performance_Profile" not in vessel_data.columns:
    profiles = ["good", "medium", "poor"]
    assigned_profiles = [profiles[i % len(profiles)] for i in range(len(vessel_data))]
    random.shuffle(assigned_profiles)
    vessel_data["Performance_Profile"] = assigned_profiles

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

            # === Default cubic generation with variation ===
            profile = vessel_data.at[idx, "Performance_Profile"]
            coeffs = performance_profiles[profile]
            profile_peaks = {"good": 125.0, "medium": 140.0, "poor": 155.0}
            target_max = profile_peaks[profile]

            raw_curve = [
                coeffs["a"] + coeffs["b"] * s + coeffs["c"] * s**2 + coeffs["d"] * s**3
                for s in speed_range
            ]
            current_at_21 = raw_curve[-1]
            scaling_factor = target_max / current_at_21
            variation = np.random.uniform(0.97, 1.03, size=len(raw_curve))
            scaled_curve = [val * scaling_factor * v for val, v in zip(raw_curve, variation)]

            df_input = pd.DataFrame({
                "Speed (knots)": speed_range,
                "Fuel Consumption (tons/day)": scaled_curve
            })

            edited_df = st.data_editor(df_input, key=f"editor_{idx}", num_rows="fixed")

            for _, row_val in edited_df.iterrows():
                s = int(row_val["Speed (knots)"])
                vessel_data.at[idx, f"Speed_{s}"] = float(row_val["Fuel Consumption (tons/day)"])

            try:
                speeds = edited_df["Speed (knots)"].values
                consumptions = edited_df["Fuel Consumption (tons/day)"].values

                poly_coeffs = np.polyfit(speeds, consumptions, deg=3)
                a, b, c, d = poly_coeffs[3], poly_coeffs[2], poly_coeffs[1], poly_coeffs[0]

                st.markdown("### 📈 Fitted Cubic Curve Coefficients:")
                st.markdown(f"**a** = {a:.3f} &nbsp;&nbsp; **b** = {b:.3f} &nbsp;&nbsp; **c** = {c:.3f} &nbsp;&nbsp; **d** = {d:.5f}")

                smooth_speeds = np.linspace(8, 21, 100)
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
                    compare_consumptions = [
                        float(compare_row.get(f"Speed_{s}", 0)) for s in speed_range
                    ]

                    # Fit cubic to comparison vessel
                    try:
                        comp_coeffs = np.polyfit(speed_range, compare_consumptions, deg=3)
                        a2, b2, c2, d2 = comp_coeffs[3], comp_coeffs[2], comp_coeffs[1], comp_coeffs[0]
                        compare_fitted = a2 + b2 * smooth_speeds + c2 * smooth_speeds**2 + d2 * smooth_speeds**3

                        fig.add_trace(go.Scatter(
                            x=smooth_speeds,
                            y=compare_fitted,
                            mode='lines',
                            name=f"{compare_vessel} (Fitted)",
                            line=dict(color='red', dash='dot')
                        ))

                    except Exception as e:
                        st.warning(f"Could not fit comparison vessel: {e}")

                fig.update_layout(
                    title="Speed vs. Fuel Consumption",
                    xaxis_title="Speed (knots)",
                    yaxis_title="Fuel Consumption (tons/day)",
                    legend=dict(x=0.01, y=0.99)
                )
                st.plotly_chart(fig, use_container_width=True)

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


# ----------------------- Compliance Section -----------------------
st.subheader("🌱 Regulatory Compliance Settings")
col1, col2 = st.columns(2)
with col1:
    required_ghg_intensity = st.number_input("Required GHG Intensity (gCO2e/MJ)", value=20)
    st.caption("Regulatory minimum GHG intensity for vessels to comply with FuelEU or other regional rules.")

with col2:
    penalty_per_excess_unit = st.number_input("Penalty per gCO2e/MJ Over Limit ($/day)", value=1000)
    st.caption("Penalty applied for each gCO2e/MJ above the required GHG intensity limit.")

# ----------------------- Chartering Sensitivity -----------------------
st.subheader("📈 Chartering Sensitivity")
chartering_sensitivity = (base_spot_rate - base_tc_rate) / base_tc_rate
st.metric(label="Chartering Sensitivity (Spot vs TC)", value=f"{chartering_sensitivity:.2%}")
st.caption("**Chartering Sensitivity**: Shows how attractive spot market rates are compared to TC rates.")

# ----------------------- Deployment Simulation Section -----------------------
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


#Voyage Simulation Advisor
st.header("🚢 Voyage Simulation Advisor")

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
