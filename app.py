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

fueleu_penalty_per_ton = st.sidebar.number_input("FuelEU Penalty (€/t CO₂eq shortfall)", min_value=0, value=240)
required_ghg_target = st.sidebar.number_input("FuelEU Required GHG Compliance (%)", min_value=0, max_value=100, value=80)

# ----------------------- MAIN PANEL -----------------------
st.title("LNG Fleet Deployment Simulator")

# Save/Load Scenario Section
with st.expander("💾 Save or Load Simulation"):
    scenario_config = {
        'scenario_name': scenario_name,
        'ets_price': ets_price,
        'lng_bunker_price': lng_bunker_price,
        'fueleu_penalty_per_ton': fueleu_penalty_per_ton,
        'required_ghg_target': required_ghg_target,
        'fleet_size_number_supply': fleet_size_number_supply,
        'fleet_size_dwt_supply_in_dwt_million': fleet_size_dwt_supply_in_dwt_million,
        'utilization_constant': utilization_constant,
        'assumed_speed': assumed_speed,
        'sea_margin': sea_margin,
        'assumed_laden_days': assumed_laden_days,
        'demand_billion_ton_mile': demand_billion_ton_mile,
        'auto_tightness': auto_tightness,
        'base_spot_rate': base_spot_rate,
        'base_tc_rate': base_tc_rate,
        'carbon_calc_method': carbon_calc_method
    }
    if st.button("Save Current Scenario"):
        st.download_button("Download JSON", data=json.dumps(scenario_config), file_name=f"{scenario_name}_scenario.json")
    uploaded_file = st.file_uploader("Upload Previous Scenario", type="json")
    if uploaded_file is not None:
        loaded_config = json.load(uploaded_file)
        st.session_state.loaded_data = loaded_config
        st.success("Scenario loaded successfully!")

# Deployment Simulation Section
st.header("Deployment Simulation Results")
results = []
for idx, vessel in vessel_data.iterrows():
    ref_total_fuel = vessel["Main_Engine_Consumption_MT_per_day"] + vessel["Generator_Consumption_MT_per_day"]
    adjusted_fuel = ref_total_fuel * (assumed_speed / assumed_speed) ** 3

    if carbon_calc_method == "Boil Off Rate":
        adjusted_fuel = vessel["Boil_Off_Rate_percent"] * vessel["Capacity_CBM"] / 1000

    auto_co2 = adjusted_fuel * 3.114
    carbon_cost = auto_co2 * ets_price
    fuel_cost = adjusted_fuel * lng_bunker_price

    compliance_gap = max(0, required_ghg_target - vessel["FuelEU_GHG_Compliance"])
    fueleu_penalty_cost = compliance_gap / 100 * auto_co2 * fueleu_penalty_per_ton

    margin_cost = vessel["Margin"]
    breakeven = fuel_cost + carbon_cost + fueleu_penalty_cost + margin_cost

    results.append({
        "Vessel": vessel["Name"],
        "Fuel Cost ($/day)": f"{fuel_cost:,.1f}",
        "Carbon Cost ($/day)": f"{carbon_cost:,.1f}",
        "FuelEU Penalty ($/day)": f"{fueleu_penalty_cost:,.1f}",
        "Margin ($/day)": f"{margin_cost:,.1f}",
        "Breakeven Spot ($/day)": f"{breakeven:,.1f}",
        "Decision": "✅ Spot Recommended" if base_spot_rate > breakeven else "❌ TC/Idle Preferred"
    })

df_result = pd.DataFrame(results)
st.dataframe(df_result.style.set_properties(**{'text-align': 'center'}).set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center')]}
]))

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
