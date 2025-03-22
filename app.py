import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

def plot_sc_curve(speeds, consumptions):
    fig, ax = plt.subplots()
    ax.plot(speeds, consumptions, marker='o', linestyle='-')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 150)
    ax.set_xlabel('Speed (knots)')
    ax.set_ylabel('Total Consumption (tons/day)')
    ax.set_title('Speed & Consumption Curve')
    ax.grid(True)
    return fig

st.set_page_config(page_title="LNG 10-Vessel Deployment Tool", layout="wide")

# ----------------------- SIDEBAR -----------------------
st.sidebar.title("⚙️ Scenario & Market Inputs")
scenario_name = st.sidebar.text_input("Scenario Name", value="My Scenario")
ets_price = st.sidebar.slider("EU ETS Carbon Price (€/t CO₂)", 60, 150, 95)
lng_bunker_price = st.sidebar.slider("LNG Bunker Price ($/ton)", 600, 1000, 730)

st.sidebar.header("💡 Freight Market Inputs")
fleet_size_number_supply = st.sidebar.number_input("Fleet Size (# of Ships)", value=3131, step=1, format="%d")
fleet_size_dwt_supply_in_dwt_million = st.sidebar.number_input("Supply (M DWT)", value=254.1, step=0.1)
utilization_constant = st.sidebar.number_input("Utilization Factor", value=0.95, step=0.01)
assumed_speed = st.sidebar.number_input("Speed (knots)", value=11.0, step=0.1)
sea_margin = st.sidebar.number_input("Sea Margin (%)", value=0.05, step=0.01)
assumed_laden_days = st.sidebar.number_input("Laden Days Fraction", value=0.4, step=0.01)
demand_billion_ton_mile = st.sidebar.number_input("Demand (Bn Ton Mile)", value=10396.0, step=10.0)
auto_tightness = st.sidebar.checkbox("Auto-calculate market tightness", value=True)

# Tightness calculation
dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
productive_laden_days_per_year = assumed_laden_days * 365
maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile

if auto_tightness:
    market_tightness = min(max(0.3 + (equilibrium / demand_billion_ton_mile), 0.0), 1.0)
else:
    market_tightness = st.sidebar.slider("Manual Tightness (0-1)", 0.0, 1.0, 0.5)

with st.sidebar.expander("🔍 Market Calculations"):
    st.markdown(f"**DWT Utilization:** {dwt_utilization:,.1f} MT")
    st.markdown(f"**Max Supply:** {maximum_supply_billion_ton_mile:,.1f} Bn Ton Mile")
    st.markdown(f"**Equilibrium:** {equilibrium:,.1f} Bn Ton Mile")
    market_status = 'Excess Supply' if equilibrium < 0 else 'Excess Demand'
    status_color = 'red' if market_status == 'Excess Supply' else 'green'
    st.markdown(f"**Market Condition:** <span style='color:{status_color}'>{market_status}</span>", unsafe_allow_html=True)
    st.markdown(f"**Tightness:** {market_tightness:.2f}")

base_spot_rate = st.sidebar.slider("Spot Rate (USD/day)", 5000, 150000, 60000, step=1000)
base_tc_rate = st.sidebar.slider("TC Rate (USD/day)", 5000, 140000, 50000, step=1000)

carbon_calc_method = st.sidebar.radio("Carbon Cost Based On", ["Main Engine Consumption", "Boil Off Rate"])

# ----------------------- MAIN PANEL -----------------------
st.title("LNG Fleet Deployment Simulator")

st.header("Vessel Profile Input")
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
        vessel_data.at[idx, "Capacity_CBM"] = st.number_input("Capacity (CBM)", value=row["Capacity_CBM"], key=f"cap_{idx}")

        vessel_data.at[idx, "Main_Engine_Consumption_MT_per_day"] = st.number_input("Main Engine (tons/day)", value=row["Main_Engine_Consumption_MT_per_day"], key=f"me_{idx}")
        vessel_data.at[idx, "Generator_Consumption_MT_per_day"] = st.number_input("Generator (tons/day)", value=row["Generator_Consumption_MT_per_day"], key=f"gen_{idx}")

        if st.toggle("More Details", key=f"toggle_{idx}"):
            with st.container():
                c1, c2 = st.columns(2)
                with c1:
                    vessel_data.at[idx, "Boil_Off_Rate_percent"] = st.number_input("Boil Off Rate (%)", value=row["Boil_Off_Rate_percent"], key=f"bor_{idx}")
                    vessel_data.at[idx, "Margin"] = st.number_input("Margin (USD/day)", value=row["Margin"], key=f"margin_{idx}")
                with c2:
                    vessel_data.at[idx, "CII_Rating"] = st.selectbox("CII Rating", options=["A", "B", "C", "D", "E"], index=["A", "B", "C", "D", "E"].index(row["CII_Rating"]), key=f"cii_{idx}")
                    vessel_data.at[idx, "FuelEU_GHG_Compliance"] = st.number_input("FuelEU GHG Intensity (%)", value=row["FuelEU_GHG_Compliance"], key=f"ghg_{idx}")
                st.markdown("---")
                st.caption("Speed & Consumption Curve")
                sc_speeds = [13, 14, 15, 16, 17, 18]
                total_consumption = [
                    row["Main_Engine_Consumption_MT_per_day"] + row["Generator_Consumption_MT_per_day"] * (speed / 17) for speed in sc_speeds
                ]
                fig = plot_sc_curve(sc_speeds, total_consumption)
                st.pyplot(fig)

# Scenario Save/Load
with st.sidebar.expander("💾 Save/Load Scenario"):
    if st.button("Save Scenario"):
        st.download_button("Download JSON", json.dumps(vessel_data.to_dict()), file_name=f"{scenario_name}.json")
    uploaded_file = st.file_uploader("Upload Saved Scenario", type=["json"])
    if uploaded_file is not None:
        uploaded_data = json.load(uploaded_file)
        st.write("Scenario loaded successfully!")
