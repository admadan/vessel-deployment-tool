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

st.title("LNG Deployment Simulator for 10 Vessels")
st.header("1️⃣ Vessel Performance Inputs")

# ----------------------- VESSEL INPUTS -----------------------
default_data = [
    {"name": f"Vessel {i+1}", "fuel": 120 if i < 5 else 140, "co2": 120 if i < 5 else 140, "bor": 0.07 if i < 5 else 0.1, "opex": 10000 if i < 5 else 12000}
    for i in range(10)
]

vessel_inputs = []

for i in range(10):
    with st.expander(f"Vessel {i+1} Inputs"):
        name = st.text_input(f"Name for Vessel {i+1}", value=default_data[i]["name"])
        fuel = st.number_input(f"Fuel Consumption (tons/day) - Vessel {i+1}", value=default_data[i]["fuel"])
        co2 = st.number_input(f"CO₂ Emissions (tons/day) - Vessel {i+1}", value=default_data[i]["co2"])
        bor = st.number_input(f"Boil-Off Rate (%/day) - Vessel {i+1}", value=default_data[i]["bor"])
        opex = st.number_input(f"OPEX (USD/day) - Vessel {i+1}", value=default_data[i]["opex"])
        vessel_inputs.append({"name": name, "fuel": fuel, "co2": co2, "bor": bor, "opex": opex})

# ----------------------- SIMULATION -----------------------
st.header("2️⃣ Simulation Results")

spot_decisions = []
breakevens = []

for vessel in vessel_inputs:
    carbon_cost = vessel["co2"] * ets_price
    fuel_cost = vessel["fuel"] * 730
    breakeven = fuel_cost + carbon_cost + vessel["opex"]
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
