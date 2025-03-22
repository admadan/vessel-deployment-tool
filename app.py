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
def sidebar_inputs():
    st.sidebar.title("⚙️ Scenario & Market Inputs")
    scenario = {
        "scenario_name": st.sidebar.text_input("Scenario Name", value=get_value("scenario_name", "My Scenario")),
        "ets_price": st.sidebar.slider("EU ETS Carbon Price (€/t CO₂)", 60, 150, get_value("ets_price", 95)),
        "lng_bunker_price": st.sidebar.slider("LNG Bunker Price ($/ton)", 600, 1000, get_value("lng_bunker_price", 730)),
        "fleet_size_number_supply": st.sidebar.number_input("Fleet Size (# of Ships)", value=get_value("fleet_size_number_supply", 3131), step=1, format="%d"),
        "fleet_size_dwt_supply_in_dwt_million": st.sidebar.number_input("Supply (M DWT)", value=get_value("fleet_size_dwt_supply_in_dwt_million", 254.1), step=0.1),
        "utilization_constant": st.sidebar.number_input("Utilization Factor", value=get_value("utilization_constant", 0.95), step=0.01),
        "assumed_speed": st.sidebar.number_input("Speed (knots)", value=get_value("assumed_speed", 11.0), step=0.1),
        "sea_margin": st.sidebar.number_input("Sea Margin (%)", value=get_value("sea_margin", 0.05), step=0.01),
        "assumed_laden_days": st.sidebar.number_input("Laden Days Fraction", value=get_value("assumed_laden_days", 0.4), step=0.01),
        "demand_billion_ton_mile": st.sidebar.number_input("Demand (Bn Ton Mile)", value=get_value("demand_billion_ton_mile", 10396.0), step=10.0),
        "auto_tightness": st.sidebar.checkbox("Auto-calculate market tightness", value=get_value("auto_tightness", True)),
        "base_spot_rate": st.sidebar.slider("Spot Rate (USD/day)", 5000, 150000, get_value("base_spot_rate", 60000), step=1000),
        "base_tc_rate": st.sidebar.slider("TC Rate (USD/day)", 5000, 140000, get_value("base_tc_rate", 50000), step=1000),
        "carbon_calc_method": st.sidebar.radio("Carbon Cost Based On", ["Main Engine Consumption", "Boil Off Rate"], index=["Main Engine Consumption", "Boil Off Rate"].index(get_value("carbon_calc_method", "Main Engine Consumption"))),
        "fueleu_penalty_per_ton": st.sidebar.number_input("FuelEU Penalty (€/t CO₂eq shortfall)", min_value=0, value=240),
        "required_ghg_target": st.sidebar.number_input("FuelEU Required GHG Compliance (%)", min_value=0, max_value=100, value=80)
    }
    return scenario

scenario = sidebar_inputs()

# ----------------------- MAIN PANEL -----------------------
st.title("LNG Fleet Deployment Simulator")

# Vessel input section
st.subheader("Vessel Input Section")
if 'vessel_data' in st.session_state.loaded_data:
    vessel_data = pd.DataFrame(st.session_state.loaded_data['vessel_data'])
else:
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
        vessel_data.at[idx, "Main_Engine_Consumption_MT_per_day"] = st.number_input("Main Engine (tons/day)", value=row["Main_Engine_Consumption_MT_per_day"], key=f"me_{idx}")
        vessel_data.at[idx, "Generator_Consumption_MT_per_day"] = st.number_input("Generator (tons/day)", value=row["Generator_Consumption_MT_per_day"], key=f"gen_{idx}")
        vessel_data.at[idx, "Boil_Off_Rate_percent"] = st.number_input("Boil Off Rate (%)", value=row["Boil_Off_Rate_percent"], key=f"bor_{idx}")
        vessel_data.at[idx, "CII_Rating"] = st.selectbox("CII Rating", options=["A", "B", "C", "D", "E"], index=["A", "B", "C", "D", "E"].index(row["CII_Rating"]), key=f"cii_{idx}")
        vessel_data.at[idx, "FuelEU_GHG_Compliance"] = st.number_input("FuelEU GHG Intensity (gCO₂eq/MJ)", value=row["FuelEU_GHG_Compliance"], key=f"ghg_{idx}")
        vessel_data.at[idx, "Margin"] = st.number_input("Margin (USD/day)", value=row["Margin"], key=f"margin_{idx}")

# Deployment simulation section
def run_deployment_simulation():
    st.subheader("Deployment Simulation Results")
    results = []
    for idx, vessel in vessel_data.iterrows():
        ref_total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]
        adjusted_fuel = ref_total_fuel * (scenario['assumed_speed'] / scenario['assumed_speed']) ** 3

        if scenario['carbon_calc_method'] == "Boil Off Rate":
            adjusted_fuel = vessel["Boil_Off_Rate_percent"] * vessel["Capacity_CBM"] / 1000

        auto_co2 = adjusted_fuel * 3.114
        carbon_cost = auto_co2 * scenario['ets_price']
        fuel_cost = adjusted_fuel * scenario['lng_bunker_price']
        compliance_gap = max(0, scenario['required_ghg_target'] - vessel["FuelEU_GHG_Compliance"])
        fueleu_penalty_cost = compliance_gap / 100 * auto_co2 * scenario['fueleu_penalty_per_ton']
        margin_cost = vessel["Margin"]
        breakeven = fuel_cost + carbon_cost + fueleu_penalty_cost + margin_cost

        results.append({
            "Vessel": vessel["Name"],
            "Fuel Cost ($/day)": f"{fuel_cost:,.1f}",
            "Carbon Cost ($/day)": f"{carbon_cost:,.1f}",
            "FuelEU Penalty ($/day)": f"{fueleu_penalty_cost:,.1f}",
            "Margin ($/day)": f"{margin_cost:,.1f}",
            "Breakeven Spot ($/day)": f"{breakeven:,.1f}",
            "Decision": "✅ Spot Recommended" if scenario['base_spot_rate'] > breakeven else "❌ TC/Idle Preferred"
        })

    df_result = pd.DataFrame(results)
    st.dataframe(df_result.style.set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ]))

run_deployment_simulation()

# Voyage simulation section
def run_voyage_simulation():
    st.subheader("Voyage Simulation Advisor")
    voyage_days = st.number_input("Voyage Duration (days)", min_value=1, value=30)
    rate_choice = st.radio("Use Spot or TC Rate for Voyage Simulation", ["Spot Rate", "TC Rate"])
    selected_rate = scenario['base_spot_rate'] if rate_choice == "Spot Rate" else scenario['base_tc_rate']

    voyage_results = []
    for index, vessel in vessel_data.iterrows():
        ref_total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]
        adjusted_fuel = ref_total_fuel * (scenario['assumed_speed'] / scenario['assumed_speed']) ** 3
        if scenario['carbon_calc_method'] == "Boil Off Rate":
            adjusted_fuel = vessel["Boil_Off_Rate_percent"] * vessel["Capacity_CBM"] / 1000

        auto_co2 = adjusted_fuel * 3.114
        carbon_cost = auto_co2 * scenario['ets_price'] * voyage_days
        fuel_cost = adjusted_fuel * scenario['lng_bunker_price'] * voyage_days
        margin_cost = vessel["Margin"] * voyage_days

        total_voyage_cost = fuel_cost + carbon_cost + margin_cost
        total_freight = selected_rate * voyage_days
        voyage_profit = total_freight - total_voyage_cost

        voyage_results.append({
            "Vessel": vessel["Name"],
            "Voyage Cost ($)": round(total_voyage_cost, 1),
            "Freight Revenue ($)": round(total_freight, 1),
            "Voyage Profit ($)": round(voyage_profit, 1)
        })

    voyage_df = pd.DataFrame(voyage_results)
    voyage_df_sorted = voyage_df.sort_values(by="Voyage Profit ($)", ascending=False)

    best_vessel = voyage_df_sorted.iloc[0]["Vessel"]

    st.dataframe(
        voyage_df_sorted.style.apply(lambda row: ["background-color: lightgreen" if row["Vessel"] == best_vessel else "" for _ in row], axis=1)
    )

    st.success(f"🚢 Recommended Vessel for this Voyage: {best_vessel}")

run_voyage_simulation()
