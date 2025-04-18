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
