import streamlit as st
import numpy as np
import pandas as pd
import json
from fpdf import FPDF

st.set_page_config(page_title="LNG 10-Vessel Deployment Tool", layout="wide")

# ----------------------- DEFAULT VESSEL DATA -----------------------
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
    "Boil_Off_Rate_percent": [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.07, 0.07, 0.07, 0.07]
})

# ----------------------- SIDEBAR INPUTS -----------------------
st.sidebar.title("⚙️ Scenario & Market Inputs")
scenario_name = st.sidebar.text_input("Scenario Name", value="My Scenario")
ets_price = st.sidebar.slider("EU ETS Carbon Price (€/t CO₂)", 60, 150, 95)
lng_bunker_price = st.sidebar.slider("LNG Bunker Price ($/ton)", 600, 1000, 730)

st.sidebar.header("💡 Freight Market Inputs")
auto_tightness = st.sidebar.checkbox("Auto-calculate market tightness", value=True)

# Market balance calculation inputs
fleet_size_number_supply = st.sidebar.number_input("Fleet Size (Number of Ships)", value=3131, step=1, format="%d")
fleet_size_dwt_supply_in_dwt_million = st.sidebar.number_input("Fleet Size Supply (Million DWT)", value=254.1, step=0.1)
utilization_constant = st.sidebar.number_input("Utilization Constant", value=0.95, step=0.01)
assumed_speed = st.sidebar.number_input("Assumed Speed (knots)", value=11.0, step=0.1)
sea_margin = st.sidebar.number_input("Sea Margin", value=0.05, step=0.01)
assumed_laden_days = st.sidebar.number_input("Assumed Laden Days Fraction", value=0.4, step=0.01)
demand_billion_ton_mile = st.sidebar.number_input("Demand (Billion Ton Mile)", value=10396.0, step=10.0)

# Auto-market tightness calculation
dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
productive_laden_days_per_year = assumed_laden_days * 365
maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile

# Tightness suggestion
if auto_tightness:
    market_tightness = min(max(0.3 + (equilibrium / demand_billion_ton_mile), 0.0), 1.0)
else:
    market_tightness = st.sidebar.slider("Manual Market Tightness (0-1)", 0.0, 1.0, 0.5)

st.sidebar.markdown(f"**Market Tightness (0-1):** {market_tightness:.2f}")

# Spot and TC rate suggestion
base_spot_rate = st.sidebar.slider("Current Spot Rate (USD/day)", 40000, 150000, 60000, step=1000)
base_tc_rate = st.sidebar.slider("Current TC Rate (USD/day)", 30000, 140000, 50000, step=1000)

# ----------------------- MAIN PANEL -----------------------
st.title("LNG Fleet Deployment Simulator")

# Editable vessel profile table
edited_df = st.data_editor(vessel_data, num_rows="dynamic", use_container_width=True)

st.header("2️⃣ Simulation Results")

spot_decisions = []
breakevens = []

for index, vessel in edited_df.iterrows():
    total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]
    auto_co2 = total_fuel * 3.17
    carbon_cost = auto_co2 * ets_price
    fuel_cost = total_fuel * lng_bunker_price
    breakeven = fuel_cost + carbon_cost + 10000  # Assuming static OPEX for now
    breakevens.append(breakeven)

    if base_spot_rate > breakeven:
        spot_decisions.append("✅ Spot Recommended")
    else:
        spot_decisions.append("❌ TC/Idle Preferred")

# ----------------------- DISPLAY RESULTS -----------------------
results = pd.DataFrame({
    "Vessel_ID": edited_df["Vessel_ID"],
    "Vessel": edited_df["Name"],
    "Breakeven Spot (USD/day)": breakevens,
    "Decision": spot_decisions
})

st.dataframe(results)

# ----------------------- MARKET BALANCE FEEDBACK -----------------------
st.subheader("🌍 Market Equilibrium Result")
result_label = "**Excess Supply**" if equilibrium < 0 else "**Excess Demand**"
st.markdown(f"Result: {result_label}  |  **Δ:** {equilibrium:,.2f} Billion Ton Miles")
