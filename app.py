import streamlit as st
import numpy as np
import pandas as pd
import json

st.set_page_config(page_title="LNG 10-Vessel Deployment Tool", layout="wide")

# ----------------------- SIDEBAR -----------------------
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = {}

# Load session values if data is loaded
loaded_data = st.session_state.loaded_data

def get_value(key, default):
    return loaded_data.get(key, default)

st.sidebar.title("⚙️ Scenario & Market Inputs")
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

base_spot_rate = st.sidebar.slider("Spot Rate (USD/day)", 5000, 150000, get_value("base_spot_rate", 60000), step=1000)
base_tc_rate = st.sidebar.slider("TC Rate (USD/day)", 5000, 140000, get_value("base_tc_rate", 50000), step=1000)
carbon_calc_method = st.sidebar.radio("Carbon Cost Based On", ["Main Engine Consumption", "Boil Off Rate"], index=["Main Engine Consumption", "Boil Off Rate"].index(get_value("carbon_calc_method", "Main Engine Consumption")))

# ----------------------- MAIN PANEL -----------------------
st.title("LNG Fleet Deployment Simulator")

st.header("Vessel Profile Input")
if 'vessel_data' in loaded_data:
    vessel_data = pd.DataFrame(loaded_data['vessel_data'])
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
                    row["Main_Engine_Consumption_MT_per_day"] + row["Generator_Consumption_MT_per_day"] * (speed / 17)
                    for speed in sc_speeds
                ]
                st.line_chart({"Speed (knots)": sc_speeds, "Total Consumption (tons/day)": total_consumption})

# Scenario Save/Load Section
with st.sidebar.expander("💾 Save/Load Scenario"):
    scenario_inputs = {
        "scenario_name": scenario_name,
        "ets_price": ets_price,
        "lng_bunker_price": lng_bunker_price,
        "fleet_size_number_supply": fleet_size_number_supply,
        "fleet_size_dwt_supply_in_dwt_million": fleet_size_dwt_supply_in_dwt_million,
        "utilization_constant": utilization_constant,
        "assumed_speed": assumed_speed,
        "sea_margin": sea_margin,
        "assumed_laden_days": assumed_laden_days,
        "demand_billion_ton_mile": demand_billion_ton_mile,
        "auto_tightness": auto_tightness,
        "base_spot_rate": base_spot_rate,
        "base_tc_rate": base_tc_rate,
        "carbon_calc_method": carbon_calc_method,
        "vessel_data": vessel_data.to_dict()
    }

    st.download_button(
        label="📥 Download Scenario JSON",
        data=json.dumps(scenario_inputs, indent=4),
        file_name=f"{scenario_name.replace(' ', '_')}.json",
        mime="application/json"
    )

    uploaded_file = st.file_uploader("📤 Upload Scenario", type=["json"])
    if uploaded_file:
        st.session_state.loaded_data = json.load(uploaded_file)
        st.experimental_rerun()



# ----------------------- Simulation Section -----------------------
st.header("Deployment Simulation Results")
with st.spinner("Calculating breakevens based on realistic speed curves..."):
    spot_decisions = []
    breakevens = []
    total_co2_emissions = []

    base_speed = 15  # reference speed where consumption is calibrated

    for index, vessel in vessel_data.iterrows():
        # Reference total consumption at base speed (15 knots)
        ref_total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]

        # Apply cubic relationship based on assumed speed
        adjusted_fuel = ref_total_fuel * (assumed_speed / base_speed) ** 3

        # If carbon calc method is Boil-Off based, switch logic
        if carbon_calc_method == "Boil Off Rate":
            adjusted_fuel = vessel["Boil_Off_Rate_percent"] * vessel["Capacity_CBM"] / 1000

        # Calculate carbon & fuel cost
        auto_co2 = adjusted_fuel * 3.114
        carbon_cost = auto_co2 * ets_price
        fuel_cost = adjusted_fuel * lng_bunker_price
        margin_cost = vessel["Margin"]
        breakeven = fuel_cost + carbon_cost + margin_cost

        breakevens.append({
            "Vessel_ID": vessel["Vessel_ID"],
            "Vessel": vessel["Name"],
            "Fuel Cost ($/day)": round(fuel_cost, 0),
            "Carbon Cost ($/day)": round(carbon_cost, 0),
            "Margin ($/day)": round(margin_cost, 0),
            "Breakeven Spot ($/day)": round(breakeven, 0)
        })

        total_co2_emissions.append(auto_co2)

        if base_spot_rate > breakeven:
            spot_decisions.append("✅ Spot Recommended")
        else:
            spot_decisions.append("❌ TC/Idle Preferred")

    results_df = pd.DataFrame(breakevens)
    results_df["Total CO₂ (t/day)"] = [f"{x:,.1f}" for x in total_co2_emissions]
    results_df["Decision"] = spot_decisions

    st.dataframe(
        results_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]}
        ])
    )

