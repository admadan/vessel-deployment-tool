# Let's generate the fully integrated dashboard script with LSTM model prediction
# for the user-selected column in the "LNG Market" section.

final_dashboard_path = "/mnt/data/lng_dashboard_with_user_selected_lstm.py"

# Full integration of LSTM model into the LNG Market section
full_lstm_integration_code = """
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set up Streamlit page config
st.set_page_config(page_title="Shipping Dashboard", layout="wide")

# Define LSTM function
def run_lstm_model(df, target_column, sequence_length=10):
    df = df.sort_values(by='Date')
    series = df[target_column].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    train_size = int(len(scaled_series) * 0.8)
    train_data = scaled_series[:train_size]
    test_data = scaled_series[train_size:]

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y_test)

    return predicted, actual

# Sidebar
page = st.sidebar.radio("Select Page", ["LNG Market (with Forecast)", "Other Pages..."])

# LNG Market Page
if page == "LNG Market (with Forecast)":
    st.title("📈 LNG Market Trends with LSTM Forecast")

    base_url = "https://docs.google.com/spreadsheets/d/1kySjcfv1jMkDRrqAD9qS10KjIs5H1Vdu/gviz/tq?tqx=out:csv&sheet="
    sheet_name = "Weekly%20data_160K%20CBM"
    data_url = f"{base_url}{sheet_name}"

    try:
        df = pd.read_csv(data_url, dtype=str)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df = df.dropna(subset=["Date"]).sort_values(by="Date")
        else:
            st.error("⚠️ 'Date' column not found in the dataset.")

        for col in df.columns:
            if col != "Date":
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        available_columns = [col for col in df.columns if col != "Date"]
        selected_column = st.selectbox("Select Parameter for LSTM Forecast", available_columns)

        start_date = st.date_input("Select Start Date", df["Date"].min())
        end_date = st.date_input("Select End Date", df["Date"].max())
        df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

        st.subheader(f"Time Series Plot for: {selected_column}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_filtered["Date"], df_filtered[selected_column], label=selected_column)
        ax.set_title(f"{selected_column} Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel(selected_column)
        ax.grid(True)
        st.pyplot(fig)

        if selected_column in df_filtered.columns and len(df_filtered) > 20:
            st.subheader("LSTM Forecast")
            predicted, actual = run_lstm_model(df_filtered, selected_column)

            fig_lstm, ax_lstm = plt.subplots(figsize=(10, 4))
            ax_lstm.plot(actual, label="Actual", color="blue")
            ax_lstm.plot(predicted, label="Predicted", color="red")
            ax_lstm.set_title(f"LSTM Prediction for {selected_column}")
            ax_lstm.set_ylabel(selected_column)
            ax_lstm.set_xlabel("Index")
            ax_lstm.legend()
            ax_lstm.grid(True)
            st.pyplot(fig_lstm)

    except Exception as e:
        st.error(f"❌ Error loading or processing data: {e}")
"""

# Save this to file
with open("LSTM_LNG_Dashboard_Output.py", "w") as f:
    f.write(full_lstm_integration_code)

final_dashboard_path
