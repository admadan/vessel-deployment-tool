import streamlit as st
import numpy as np
import pandas as pd
import json

# ----------------------- MAIN PANEL -----------------------
st.title("LNG Fleet Deployment Simulator")

# Vessel Data
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

cols = st.columns(2)
for idx, row in vessel_data.iterrows():
    with cols[idx % 2].expander(f"🚢 {row['Name']}"):
        vessel_data.at[idx, "Name"] = st.text_input("Vessel Name", value=row["Name"], key=f"name_{idx}")
        vessel_data.at[idx, "Length_m"] = st.number_input("Length (m)", value=row["Length_m"], key=f"len_{idx}")
        vessel_data.at[idx, "Beam_m"] = st.number_input("Beam (m)", value=row["Beam_m"], key=f"beam_{idx}")
        vessel_data.at[idx, "Draft_m"] = st.number_input("Draft (m)", value=row["Draft_m"], key=f"draft_{idx}")
        vessel_data.at[idx, "Margin"] = st.number_input("Margin (USD/day)", value=row["Margin"], key=f"margin_{idx}")

        if st.toggle("Performance Details", key=f"toggle_{idx}"):
            with st.container():
                st.caption("Speed & Consumption Curve (auto-adjusted to Freight Market speed input)")
                compare_toggle = st.checkbox("Compare with another vessel", key=f"compare_toggle_{idx}")
                min_speed = max(8, assumed_speed - 3)
                max_speed = min(20, assumed_speed + 3)
                speed_range = list(range(int(min_speed), int(max_speed) + 1))
                base_speed = assumed_speed

                ref_total_consumption = row["Main_Engine_Consumption_MT_per_day"] + row["Generator_Consumption_MT_per_day"]
                total_consumption = [ref_total_consumption * (speed / base_speed) ** 3 for speed in speed_range]
                df_curve = pd.DataFrame({"Speed (knots)": speed_range, row["Name"]: total_consumption}).set_index("Speed (knots)")

                if compare_toggle:
                    compare_vessel = st.selectbox("Select vessel to compare", [v for i, v in enumerate(vessel_data['Name']) if i != idx], key=f"compare_{idx}")
                    compare_row = vessel_data[vessel_data['Name'] == compare_vessel].iloc[0]
                    compare_ref_consumption = compare_row["Main_Engine_Consumption_MT_per_day"] + compare_row["Generator_Consumption_MT_per_day"]
                    compare_total_consumption = [compare_ref_consumption * (speed / base_speed) ** 3 for speed in speed_range]
                    df_curve[compare_vessel] = compare_total_consumption

                st.line_chart(df_curve)

                vessel_data.at[idx, "Main_Engine_Consumption_MT_per_day"] = st.number_input("Main Engine (tons/day)", value=row["Main_Engine_Consumption_MT_per_day"], key=f"me_{idx}")
                vessel_data.at[idx, "Generator_Consumption_MT_per_day"] = st.number_input("Generator (tons/day)", value=row["Generator_Consumption_MT_per_day"], key=f"gen_{idx}")
                c1, c2 = st.columns(2)
                with c1:
                    vessel_data.at[idx, "Boil_Off_Rate_percent"] = st.number_input("Boil Off Rate (%)", value=row["Boil_Off_Rate_percent"], key=f"bor_{idx}")
                with c2:
                    vessel_data.at[idx, "CII_Rating"] = st.selectbox("CII Rating", options=["A", "B", "C", "D", "E"], index=["A", "B", "C", "D", "E"].index(row["CII_Rating"]), key=f"cii_{idx}")
                    vessel_data.at[idx, "FuelEU_GHG_Compliance"] = st.number_input("FuelEU GHG Intensity (%)", value=row["FuelEU_GHG_Compliance"], key=f"ghg_{idx}")

# Deployment Simulation Section
st.header("Deployment Simulation Results")

spot_decisions = []
breakevens = []
total_co2_emissions = []
for index, vessel in vessel_data.iterrows():
    ref_total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]
    adjusted_fuel = ref_total_fuel * (assumed_speed / assumed_speed) ** 3
    if carbon_calc_method == "Boil Off Rate":
        adjusted_fuel = vessel["Boil_Off_Rate_percent"] * vessel["Capacity_CBM"] / 1000

    auto_co2 = adjusted_fuel * 3.114
    carbon_cost = auto_co2 * ets_price
    fuel_cost = adjusted_fuel * lng_bunker_price
    margin_cost = vessel["Margin"]
    breakeven = fuel_cost + carbon_cost + margin_cost

    breakevens.append({
        "Vessel_ID": vessel["Vessel_ID"],
        "Vessel": vessel["Name"],
        "Fuel Cost ($/day)": f"{fuel_cost:,.1f}",
        "Carbon Cost ($/day)": f"{carbon_cost:,.1f}",
        "Margin ($/day)": f"{margin_cost:,.1f}",
        "Breakeven Spot ($/day)": f"{breakeven:,.1f}"
    })

    total_co2_emissions.append(f"{auto_co2:,.1f}")
    spot_decisions.append("✅ Spot Recommended" if base_spot_rate > breakeven else "❌ TC/Idle Preferred")

results_df = pd.DataFrame(breakevens)
results_df["Total CO₂ (t/day)"] = total_co2_emissions
results_df["Decision"] = spot_decisions
st.dataframe(results_df)

# Voyage Simulation Section
st.header("Voyage Simulation Advisor")
voyage_days = st.number_input("Voyage Duration (days)", min_value=1, value=30)
rate_choice = st.radio("Use Spot or TC Rate for Voyage Simulation", ["Spot Rate", "TC Rate"])
selected_rate = base_spot_rate if rate_choice == "Spot Rate" else base_tc_rate

voyage_results = []
for index, vessel in vessel_data.iterrows():
    ref_total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]
    adjusted_fuel = ref_total_fuel * (assumed_speed / assumed_speed) ** 3
    if carbon_calc_method == "Boil Off Rate":
        adjusted_fuel = vessel["Boil_Off_Rate_percent"] * vessel["Capacity_CBM"] / 1000

    auto_co2 = adjusted_fuel * 3.114
    carbon_cost = auto_co2 * ets_price * voyage_days
    fuel_cost = adjusted_fuel * lng_bunker_price * voyage_days
    margin_cost = vessel["Margin"] * voyage_days

    total_voyage_cost = fuel_cost + carbon_cost + margin_cost
    total_freight = selected_rate * voyage_days
    voyage_profit = total_freight - total_voyage_cost

    voyage_results.append({
        "Vessel": vessel["Name"],
        "Voyage Cost ($)": f"{total_voyage_cost:,.1f}",
        "Freight Revenue ($)": f"{total_freight:,.1f}",
        "Voyage Profit ($)": f"{voyage_profit:,.1f}"
    })

voyage_df = pd.DataFrame(voyage_results)
voyage_df_sorted = voyage_df.sort_values(by="Voyage Profit ($)", ascending=False)
st.dataframe(voyage_df_sorted)
best_vessel = voyage_df_sorted.iloc[0]["Vessel"]
st.success(f"🚢 Recommended Vessel for this Voyage: {best_vessel}")
