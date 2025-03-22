import streamlit as st
import numpy as np
import pandas as pd
import json
from fpdf import FPDF

st.set_page_config(page_title="LNG 10-Vessel Deployment Tool", layout="wide")

# ----------------------- SIDEBAR -----------------------
st.sidebar.title("⚙️ Scenario & Market Inputs")
scenario_name = st.sidebar.text_input("Scenario Name", value="My Scenario")
ets_price = st.sidebar.slider("EU ETS Carbon Price (€/t CO₂)", 60, 150, 95)
lng_bunker_price = st.sidebar.slider("LNG Bunker Price ($/ton)", 600, 1000, 730)

st.sidebar.header("💡 Freight Market Inputs")
auto_tightness = st.sidebar.checkbox("Auto-calculate market tightness", value=True)

fleet_size_number_supply = st.sidebar.number_input("Fleet Size (Number of Ships)", value=3131, step=1, format="%d")
fleet_size_dwt_supply_in_dwt_million = st.sidebar.number_input("Fleet Size Supply (Million DWT)", value=254.1, step=0.1)
utilization_constant = st.sidebar.number_input("Utilization Constant", value=0.95, step=0.01)
assumed_speed = st.sidebar.number_input("Assumed Speed (knots)", value=11.0, step=0.1)
sea_margin = st.sidebar.number_input("Sea Margin", value=0.05, step=0.01)
assumed_laden_days = st.sidebar.number_input("Assumed Laden Days Fraction", value=0.4, step=0.01)
demand_billion_ton_mile = st.sidebar.number_input("Demand (Billion Ton Mile)", value=10396.0, step=10.0)

# Tightness calculation
dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
productive_laden_days_per_year = assumed_laden_days * 365
maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile

if auto_tightness:
    market_tightness = min(max(0.3 + (equilibrium / demand_billion_ton_mile), 0.0), 1.0)
else:
    market_tightness = st.sidebar.slider("Manual Market Tightness (0-1)", 0.0, 1.0, 0.5)

st.sidebar.markdown(f"**Market Tightness (0-1):** {market_tightness:.2f}")

base_spot_rate = st.sidebar.slider("Current Spot Rate (USD/day)", 5000, 150000, 60000, step=1000)
base_tc_rate = st.sidebar.slider("Current TC Rate (USD/day)", 5000, 140000, 50000, step=1000)

# ----------------------- MAIN PANEL -----------------------
st.title("LNG Fleet Deployment Simulator")

st.header("Vessel Profile Input")
vessel_data = pd.DataFrame({
    "Vessel_ID": range(1, 11),
    "Name": [
        "LNG Carrier Alpha", "LNG Carrier Beta", "LNG Carrier Gamma", "LNG Carrier Delta",
        "LNG Carrier Epsilon", "LNG Carrier Zeta", "LNG Carrier Theta", "LNG Carrier Iota",
        "LNG Carrier Kappa", "LNG Carrier Lambda"
    ],
    "Capacity_CBM": [160000] * 10,
    "FuelEU_GHG_Compliance": [65, 65, 65, 80, 80, 80, 95, 95, 95, 95],
    "CII_Rating": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
    "Main_Engine_Consumption_MT_per_day": [70, 72, 74, 85, 88, 90, 100, 102, 105, 107],
    "Generator_Consumption_MT_per_day": [5, 5, 5, 6, 6, 6, 7, 7, 7, 7],
    "Boil_Off_Rate_percent": [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.07, 0.07, 0.07, 0.07],
    "Margin": [2000] * 10
})

for idx, row in vessel_data.iterrows():
    with st.expander(f"{row['Name']}"):
        col1, col2 = st.columns(2)
        with col1:
            vessel_data.at[idx, "Main_Engine_Consumption_MT_per_day"] = st.number_input(f"Main Engine Consumption (tons/day) - {row['Name']}", value=row["Main_Engine_Consumption_MT_per_day"], key=f"me_{idx}")
        with col2:
            vessel_data.at[idx, "Generator_Consumption_MT_per_day"] = st.number_input(f"Generator Consumption (tons/day) - {row['Name']}", value=row["Generator_Consumption_MT_per_day"], key=f"gen_{idx}")

        show_more = st.toggle("Show More Details", key=f"toggle_{idx}")
        if show_more:
            vessel_data.at[idx, "Boil_Off_Rate_percent"] = st.number_input(f"Boil Off Rate (%/day) - {row['Name']}", value=row["Boil_Off_Rate_percent"], key=f"bor_{idx}")
            vessel_data.at[idx, "Margin"] = st.number_input(f"Margin (USD/day) - {row['Name']}", value=row["Margin"], key=f"margin_{idx}")

# ----------------------- Simulation Section -----------------------
st.header("2️⃣ Simulation Results")

spot_decisions = []
breakevens = []
total_co2_emissions = []

for index, vessel in vessel_data.iterrows():
    total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]
    auto_co2 = total_fuel * 3.114
    carbon_cost = auto_co2 * ets_price
    fuel_cost = total_fuel * lng_bunker_price
    margin_cost = vessel["Margin"]
    breakeven = fuel_cost + carbon_cost + margin_cost

    breakevens.append({
        "Vessel_ID": vessel["Vessel_ID"],
        "Vessel": vessel["Name"],
        "Fuel Cost": fuel_cost,
        "Carbon Cost": carbon_cost,
        "Margin": margin_cost,
        "Breakeven Spot (USD/day)": breakeven
    })

    total_co2_emissions.append(auto_co2)

    if base_spot_rate > breakeven:
        spot_decisions.append("✅ Spot Recommended")
    else:
        spot_decisions.append("❌ TC/Idle Preferred")

results_df = pd.DataFrame(breakevens)
results_df["Decision"] = spot_decisions
results_df["Total CO₂ (t/day)"] = total_co2_emissions

st.dataframe(results_df)

# ----------------------- Save / Load Scenarios -----------------------
if "saved_scenarios" not in st.session_state:
    st.session_state["saved_scenarios"] = {}

if st.button("💾 Save Scenario"):
    st.session_state["saved_scenarios"][scenario_name] = vessel_data.to_dict()
    st.success(f"Scenario '{scenario_name}' saved.")

if st.session_state["saved_scenarios"]:
    st.subheader("Saved Scenarios")
    for key in st.session_state["saved_scenarios"]:
        st.text(f"- {key}")

# ----------------------- Market Balance Feedback -----------------------
st.subheader("🌍 Market Equilibrium Result")
result_label = "**Excess Supply**" if equilibrium < 0 else "**Excess Demand**"
st.markdown(f"Result: {result_label}  |  **Δ:** {equilibrium:,.2f} Billion Ton Miles")
