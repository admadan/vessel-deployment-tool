import streamlit as st
import numpy as np
import pandas as pd
import json
from fpdf import FPDF

st.set_page_config(page_title="LNG 10-Vessel Deployment Tool", layout="wide")

# ----------------------- SIDEBAR INPUTS -----------------------
st.sidebar.title("⚙️ Global Scenario Inputs")
scenario_name = st.sidebar.text_input("Scenario Name", value="My Scenario")
ets_price = st.sidebar.slider("EU ETS Carbon Price (€/t CO₂)", 60, 150, 95)
base_spot_rate = st.sidebar.slider("Base Spot Rate (USD/day)", 40000, 120000, 60000)
demand_level = st.sidebar.slider("Market Tightness (Demand)", 50, 200, 100)
lng_bunker_price = st.sidebar.slider("LNG Bunker Price ($/ton)", 600, 1000, 730)

# ----------------------- GLOBAL MARKET CALCULATIONS -----------------------
st.sidebar.title("🌍 Market Balance Inputs")
fleet_size_number_supply = st.sidebar.number_input("Fleet Size (Number of Ships)", value=3131, step=1, format="%d")
fleet_size_dwt_supply_in_dwt_million = st.sidebar.number_input("Fleet Size Supply (Million DWT)", value=254.1, step=0.1)
utilization_constant = st.sidebar.number_input("Utilization Constant", value=0.95, step=0.01)
assumed_speed = st.sidebar.number_input("Assumed Speed (knots)", value=11.0, step=0.1)
sea_margin = st.sidebar.number_input("Sea Margin", value=0.05, step=0.01)
assumed_laden_days = st.sidebar.number_input("Assumed Laden Days Fraction", value=0.4, step=0.01)
demand_billion_ton_mile = st.sidebar.number_input("Demand (Billion Ton Mile)", value=10396.0, step=10.0)

dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
productive_laden_days_per_year = assumed_laden_days * 365
maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile
result = "Excess Supply" if equilibrium < 0 else "Excess Demand"

st.sidebar.markdown(f"**Balance Result:** {result}")
st.sidebar.markdown(f"**Supply-Demand Delta:** {equilibrium:,.2f} Billion Ton Miles")

st.title("LNG Deployment Simulator for 10 Vessels")
st.header("1️⃣ Vessel Performance Inputs")

# ----------------------- VESSEL INPUTS -----------------------
default_data = [
    {"name": f"Vessel {i+1}", "fuel": 120 if i < 5 else 140, "aux": 5, "bor": 0.07 if i < 5 else 0.1, "opex": 10000 if i < 5 else 12000, "margin": 2000}
    for i in range(10)
]

vessel_inputs = []

for i in range(10):
    with st.expander(f"Vessel {i+1} Inputs"):
        name = st.text_input(f"Name for Vessel {i+1}", value=default_data[i]["name"])
        fuel = st.number_input(f"Main Engine Fuel Consumption (tons/day) - Vessel {i+1}", value=default_data[i]["fuel"])
        aux = st.number_input(f"Auxiliary Engine Load (tons/day) - Vessel {i+1}", value=default_data[i]["aux"])
        bor = st.number_input(f"Boil-Off Rate (%/day) - Vessel {i+1}", value=default_data[i]["bor"])
        opex = st.number_input(f"OPEX (USD/day) - Vessel {i+1}", value=default_data[i]["opex"])
        margin = st.number_input(f"Maintenance Margin (USD/day) - Vessel {i+1}", value=default_data[i]["margin"])

        total_fuel = fuel + aux
        auto_co2 = total_fuel * 3.17

        st.info(f"Auto-calculated CO₂ Emissions: {auto_co2:.2f} tons/day")

        vessel_inputs.append({"name": name, "fuel": fuel, "aux": aux, "bor": bor, "opex": opex, "margin": margin, "co2": auto_co2})

# ----------------------- SIMULATION -----------------------
st.header("2️⃣ Simulation Results")

spot_decisions = []
breakevens = []

for vessel in vessel_inputs:
    carbon_cost = vessel["co2"] * ets_price
    fuel_cost = (vessel["fuel"] + vessel["aux"]) * lng_bunker_price
    breakeven = fuel_cost + carbon_cost + vessel["opex"] + vessel["margin"]
    breakevens.append(breakeven)

    if base_spot_rate > breakeven:
        spot_decisions.append("✅ Spot Recommended")
    else:
        spot_decisions.append("❌ TC/Idle Preferred")

# ----------------------- DISPLAY RESULTS -----------------------
results = pd.DataFrame({
    "Vessel": [v["name"] for v in vessel_inputs],
    "Breakeven Spot (USD/day)": breakevens,
    "Decision": spot_decisions
})

st.dataframe(results)

# ----------------------- SAVE SCENARIO -----------------------
st.sidebar.header("💾 Save/Load Scenario")
if st.sidebar.button("Save Scenario"):
    with open(f"{scenario_name}_scenario.json", "w") as f:
        json.dump(vessel_inputs, f)
    st.sidebar.success("Scenario Saved!")

uploaded_file = st.sidebar.file_uploader("Load Scenario", type="json")
if uploaded_file is not None:
    vessel_inputs = json.load(uploaded_file)
    st.sidebar.success("Scenario Loaded!")

# ----------------------- EXPORT PDF -----------------------
if st.button("📄 Export Report as PDF"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="LNG Fleet Deployment Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Scenario: {scenario_name}", ln=True)
    pdf.cell(200, 10, txt=f"ETS Price: €{ets_price}/t CO₂", ln=True)

    for i, vessel in enumerate(vessel_inputs):
        pdf.cell(200, 10, txt=f"{vessel['name']} | Breakeven: ${breakevens[i]:,.0f} | {spot_decisions[i]}", ln=True)

    pdf.output(f"{scenario_name}_deployment_report.pdf")
    st.success("PDF Report Generated!")
