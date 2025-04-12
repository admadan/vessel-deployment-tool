import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Set the page configuration to wide mode by default
st.set_page_config(layout="wide")

st.sidebar.title("Navigation")
show_lng_market_button = st.sidebar.button("LNG Market")
show_lng_prediction_button = st.sidebar.button("LNG Market Prediction") # Added prediction button

if "show_lng_market" not in st.session_state:
    st.session_state["show_lng_market"] = False
if "show_lng_prediction" not in st.session_state:
    st.session_state["show_lng_prediction"] = False

if show_lng_market_button:
    st.session_state["show_lng_market"] = True
    st.session_state["show_lng_prediction"] = False
elif show_lng_prediction_button:
    st.session_state["show_lng_market"] = False
    st.session_state["show_lng_prediction"] = True

if st.session_state["show_lng_market"]:
    st.title("📈 LNG Market Trends")

   

    base_url = "https://docs.google.com/spreadsheets/d/1kySjcfv1jMkDRrqAD9qS10KjIs5H1Vdu/gviz/tq?tqx=out:csv&sheet="
    sheet_names = {
        "Weekly": "Weekly%20data_160K%20CBM",
        "Monthly": "Monthly%20data_160K%20CBM",
        "Yearly": "Yearly%20data_160%20CBM"
    }

    freq_option = st.radio("Select Data Frequency", ["Weekly", "Monthly", "Yearly"])
    google_sheets_url = f"{base_url}{sheet_names[freq_option]}"

    df_filtered = pd.DataFrame()

    try:
        df_selected = pd.read_csv(google_sheets_url, dtype=str)

        if "Date" in df_selected.columns:
            df_selected["Date"] = pd.to_datetime(df_selected["Date"], errors='coerce')
            df_selected = df_selected.dropna(subset=["Date"]).sort_values(by="Date")
        else:
            st.error("⚠ 'Date' column not found in the dataset.")

        for col in df_selected.columns:
            if col != "Date":
                df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0)

        available_columns = [col for col in df_selected.columns if col != "Date"]
        column_options = st.multiselect("Select Data Columns", available_columns, default=available_columns[:2] if available_columns else [])

        if "Date" in df_selected.columns:
            start_date = st.date_input("Select Start Date", df_selected["Date"].min())
            end_date = st.date_input("Select End Date", df_selected["Date"].max())
            df_filtered = df_selected[(df_selected["Date"] >= pd.to_datetime(start_date)) & (df_selected["Date"] <= pd.to_datetime(end_date))]

            
            if len(column_options) > 0:
                num_plots = (len(column_options) + 1) // 2
                specs = [[{'secondary_y': True}] for _ in range(num_plots)]
                fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.3, specs=specs)

                for i in range(0, len(column_options), 2):
                    row_num = (i // 2) + 1

                    # Plot the first column in the pair
                    fig.add_trace(
                        go.Scatter(
                            x=df_filtered["Date"],
                            y=df_filtered[column_options[i]],
                            mode='lines',
                            name=column_options[i],
                            hovertemplate='Date: %{x}<br>Value: %{y}<extra></extra>',
                            showlegend=True,
                            legendgroup=column_options[i],
                        ),
                        row=row_num,
                        col=1,
                        secondary_y=False,
                    )
                    fig.update_yaxes(title_text=column_options[i], row=row_num, col=1, secondary_y=False)

                    # Plot the second column in the pair (if it exists)
                    if i + 1 < len(column_options):
                        fig.add_trace(
                            go.Scatter(
                                x=df_filtered["Date"],
                                y=df_filtered[column_options[i + 1]],
                                mode='lines',
                                name=column_options[i + 1],
                                hovertemplate='Date: %{x}<br>Value: %{y}<extra></extra>',
                                showlegend=True,
                                legendgroup=column_options[i + 1],
                            ),
                            row=row_num,
                            col=1,
                            secondary_y=True,
                        )
                        fig.update_yaxes(title_text=column_options[i + 1], row=row_num, col=1, secondary_y=True)

                

                fig.update_layout(
                    title="LNG Market Rates Over Time",
                    xaxis=dict(
                        title=f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                        tickangle=45,
                        tickformatstops=[dict(dtickrange=[None, None], value="%Y")],
                        range=[df_filtered["Date"].min(), df_filtered["Date"].max()]
                    ),
                    hovermode="x unified",
                    showlegend=True,  # Set showlegend to False at the layout level
                    height=300 * num_plots,
                )






                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Please select at least one data column.")

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")

elif st.session_state["show_lng_prediction"]:
    st.title("🔮 LNG Market Prediction")

    base_url = "https://docs.google.com/spreadsheets/d/1kySjcfv1jMkDRrqAD9qS10KjIs5H1Vdu/gviz/tq?tqx=out:csv&sheet="
    sheet_names = {
        "Weekly": "Weekly%20data_160K%20CBM",
        "Monthly": "Monthly%20data_160K%20CBM",
        "Yearly": "Yearly%20data_160%20CBM"
    }

    freq_option = st.radio("Select Data Frequency", ["Weekly", "Monthly", "Yearly"])
    google_sheets_url = f"{base_url}{sheet_names[freq_option]}"

    df = pd.DataFrame()
    available_cols = []  # Define with a default empty list

    try:
        st.subheader("📊 Data Loading and Initial Processing")
        df = pd.read_csv(google_sheets_url, dtype=str)
        st.write(f"Shape after loading: {df.shape}")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df = df.dropna(subset=["Date"]).sort_values(by="Date")
            st.write(f"Shape after processing 'Date' column: {df.shape}")
            if df.empty:
                st.error("⚠ No data after loading and processing 'Date'.")
                raise ValueError("No data after initial processing.")
        else:
            st.error("⚠ 'Date' column not found.")
            raise ValueError("'Date' column not found.")

        for col in df.columns:
            if col != "Date":
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        st.write(f"Shape after converting to numeric: {df.shape}")

        available_cols = [col for col in df.columns if col != "Date"] # Define after successful loading

        st.subheader("⚙️ Feature Engineering")
        target_variable_option = st.selectbox("Select Target Variable:", available_cols)
        market_features_options = st.multiselect("Select Market Features:", available_cols)
        sequence_length = st.slider("Sequence Length:", min_value=5, max_value=60,
                                    value=10 if freq_option in ["Weekly", "Monthly"] else 15, step=5)

        future_weeks = None
        if freq_option == "Weekly":
            future_weeks = st.number_input("Future Weeks:", min_value=1, max_value=104, value=54, step=1)
        elif freq_option == "Monthly":
            future_weeks = st.number_input("Future Months:", min_value=1, max_value=24, value=12, step=1)
        elif freq_option == "Yearly":
            future_weeks = st.number_input("Future Years:", min_value=1, max_value=5, value=3, step=1)

        TARGET_VARIABLE = target_variable_option
        MARKET_FEATURES = market_features_options

        if TARGET_VARIABLE and MARKET_FEATURES and future_weeks is not None:
            if TARGET_VARIABLE in df.columns and all(feature in df.columns for feature in MARKET_FEATURES):
                # Initial dropna focusing on the target variable and date
                cols_to_check_initial_dropna = [TARGET_VARIABLE] + ['Date']
                df.dropna(subset=cols_to_check_initial_dropna, inplace=True)
                st.write(f"Shape after initial dropna (target, date): {df.shape}")

                # Imputation for market features (with caution)
                if MARKET_FEATURES:
                    df[MARKET_FEATURES] = df[MARKET_FEATURES].fillna(method='ffill')
                    st.write(f"Shape after market feature imputation: {df.shape}")

                # Feature Engineering
                df['Lag_1'] = df[TARGET_VARIABLE].shift(1)
                df['Lag_2'] = df[TARGET_VARIABLE].shift(2)
                df['Lag_3'] = df[TARGET_VARIABLE].shift(3)
                df['MA_3'] = df[TARGET_VARIABLE].rolling(window=3).mean()
                df['MA_7'] = df[TARGET_VARIABLE].rolling(window=7).mean()
                df['ROC_1'] = (df[TARGET_VARIABLE] - df['Lag_1']) / (df['Lag_1'] + 1e-8)
                df['ROC_2'] = (df[TARGET_VARIABLE] - df['Lag_2']) / (df['Lag_2'] + 1e-8)
                # Add more lags for yearly data
                if freq_option == "Yearly":
                    df['Lag_4'] = df[TARGET_VARIABLE].shift(4)
                    df['Lag_5'] = df[TARGET_VARIABLE].shift(5)

                for col in MARKET_FEATURES:
                    df[f'{col}_Lag_1'] = df[col].shift(1)
                    df[f'{col}_MA_4'] = df[col].rolling(window=4).mean()
                    df[f'{col}_ROC'] = df[col].pct_change()

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                st.write(f"Shape after feature engineering: {df.shape}")

                # Determine rows to drop
                rows_to_drop = max([1 if 'Lag' in col else 0 for col in df.columns] +
                                   [2 if 'MA_3' in col else 0 for col in df.columns] +
                                   [6 if 'MA_7' in col else 0 for col in df.columns] +
                                   [3 if '_MA_4' in col else 0 for col in df.columns] +
                                   [1 if 'ROC' in col and '_ROC' not in col else 0 for col in df.columns] +
                                   ([4, 5] if freq_option == "Yearly" else [0])) # Account for extra yearly lags

                df_trimmed = df.iloc[rows_to_drop:].copy()
                st.write(f"Shape after trimming initial NaN rows: {df_trimmed.shape}")

                df_trimmed.dropna(inplace=True)
                st.write(f"Shape after final NaN removal: {df_trimmed.shape}")

                if df_trimmed.empty:
                    st.error("⚠ No data remaining after final NaN removal.")
                    raise ValueError("No data after final NaN removal.")

                feature_cols = [col for col in df_trimmed.columns if col not in ['Date'] and col != TARGET_VARIABLE] + [TARGET_VARIABLE]

                if feature_cols and TARGET_VARIABLE in df_trimmed.columns:
                    if df_trimmed[feature_cols].empty:
                        st.error("⚠ No valid features to scale.")
                        raise ValueError("No valid features for scaling.")

                    scaler = MinMaxScaler()
                    try:
                        data_scaled = scaler.fit_transform(df_trimmed[feature_cols])
                        st.write(f"Shape after scaling: {data_scaled.shape} (number of samples, number of features)")
                    except ValueError as e:
                        st.error(f"⚠ Error during scaling: {e}")
                        raise e

                    def create_sequences(data, seq_len):
                        X, y = [], []
                        for i in range(len(data) - seq_len):
                            X.append(data[i:i+seq_len])
                            y.append(data[i+seq_len][-1])
                        return np.array(X), np.array(y)

                    if len(data_scaled) <= sequence_length:
                        st.error(f"⚠ Insufficient data ({len(data_scaled)} samples) for sequence length {sequence_length}.")
                        raise ValueError("Insufficient data for sequence generation.")

                    train_ratio = 0.9
                    train_size = int(len(data_scaled) * train_ratio)
                    train_data = data_scaled[:train_size]
                    test_data = data_scaled[train_size - sequence_length:]

                    if len(train_data) == 0 or len(test_data) == 0:
                        st.error("⚠ Not enough data for train/test split.")
                        raise ValueError("Insufficient data for train/test split.")

                    X_train, y_train = create_sequences(train_data, sequence_length)
                    X_test, y_test = create_sequences(test_data, sequence_length)
                    st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                    st.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

                    if np.isnan(X_train).any() or np.isinf(X_train).any() or np.isnan(y_train).any() or np.isinf(y_train).any():
                        st.error("⚠ NaN or Inf found in training data.")
                        raise ValueError("NaN or Inf in training data.")
                    if np.isnan(X_test).any() or np.isinf(X_test).any() or np.isnan(y_test).any() or np.isinf(y_test).any():
                        st.error("⚠ NaN or Inf found in testing data.")
                        raise ValueError("NaN or Inf in testing data.")

                    # === LSTM Model Training ===
                    model = Sequential([
                        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                        Dropout(0.3),
                        LSTM(128),
                        Dropout(0.3),
                        Dense(64, activation='relu', kernel_regularizer=l2(0.01)) if freq_option == "Yearly" else Dense(1, kernel_regularizer=l2(0.01)),
                        Dense(1, kernel_regularizer=l2(0.01))
                    ])

                    model.compile(optimizer=Adam(learning_rate=0.0005 if freq_option == "Yearly" else 0.001), loss='mean_squared_error')
                    early_stop = EarlyStopping(monitor='val_loss', patience=20 if freq_option == "Yearly" else 15, restore_best_weights=True)
                    history = model.fit(X_train, y_train, epochs=200 if freq_option == "Yearly" else 100, batch_size=16 if freq_option == "Yearly" else 32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

                    # === Inverse Transform and Evaluate ===
                    target_scaler = MinMaxScaler()
                    target_scaler.fit(df_trimmed[[TARGET_VARIABLE]]) # Fit on the trimmed data

                    try:
                        y_train_pred = model.predict(X_train, verbose=0)
                        y_test_pred = model.predict(X_test, verbose=0)
                    except ValueError as e:
                        st.error(f"⚠ Error during model prediction: {e}")
                        raise e

                    train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))
                    test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
                    train_pred = target_scaler.inverse_transform(y_train_pred)
                    test_pred = target_scaler.inverse_transform(y_test_pred)

                    # === Forecast Future ===
                    last_sequence = X_test[-1]
                    future_preds = []
                    for _ in range(future_weeks):
                        try:
                            next_pred = model.predict(last_sequence.reshape(1, sequence_length, -1), verbose=0)
                        except ValueError as e:
                            st.error(f"⚠ Error during future prediction: {e}")
                            raise e
                        future_preds.append(next_pred[0, 0])
                        new_step = np.copy(last_sequence[-1])
                        new_step[-1] = next_pred
                        last_sequence = np.vstack([last_sequence[1:], new_step])

                    future_preds = np.array(future_preds).reshape(-1, 1)
                    future_inv = target_scaler.inverse_transform(future_preds)

                    start_date = df_trimmed['Date'].iloc[-1] + pd.Timedelta(days=1) # Use trimmed date
                    if freq_option == "Weekly":
                        future_dates = pd.date_range(start=start_date, periods=future_weeks, freq='W')
                    elif freq_option == "Monthly":
                        future_dates = pd.date_range(start=start_date, periods=future_weeks, freq='M')
                    elif freq_option == "Yearly":
                        future_dates = pd.date_range(start=start_date, periods=future_weeks, freq='Y')

                    # === Display Future Predictions ===
                    st.subheader(f"📈 Future Predictions ({freq_option[:-4] if freq_option != 'Yearly' else 'Yearly'}):")
                    predictions_df = pd.DataFrame({'Date': future_dates, f'Predicted {TARGET_VARIABLE}': future_inv.flatten()})
                    st.dataframe(predictions_df)

                    # === Metrics ===
                    def print_metrics(y_true, y_pred, label):
                        mse = mean_squared_error(y_true, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_true, y_pred)
                        r2 = r2_score(y_true, y_pred)
                        st.write(f"\n**{label} Performance:**")
                        st.write(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
                        return rmse, r2

                    rmse_test, r2_test = print_metrics(test_actual, test_pred, "Test")
                    rmse_train, r2_train = print_metrics(train_actual, train_pred, "Train")

                    # === Plot ===
                    plt.figure(figsize=(16, 6))
                    date_index = df_trimmed['Date'].iloc[sequence_length:].reset_index(drop=True) # Use trimmed date

                    train_dates = date_index[:len(train_actual)]
                    test_dates = date_index[len(train_actual):len(train_actual) + len(test_actual)]

                    plt.plot(train_dates, train_actual, label='Train Actual')
                    plt.plot(train_dates, train_pred, label='Train Predicted')
                    plt.plot(test_dates, test_actual, label='Test Actual')
                    plt.plot(test_dates, test_pred, label='Test Predicted')
                    plt.plot(future_dates, future_inv, label=f'{future_weeks}-{freq_option[:-4] if freq_option != "Yearly" else "Year"} Forecast', linestyle='--', color='purple')
                    plt.fill_between(future_dates,
                                     (future_inv - rmse_test).flatten(),
                                     (future_inv + rmse_test).flatten(),
                                     color='violet', alpha=0.2, label=f'±1 RMSE (±{rmse_test:.0f})')

                    plt.title(f"LSTM Forecast for {TARGET_VARIABLE}\nR² Test: {r2_test:.4f}")
                    plt.xlabel("Date")
                    plt.ylabel(TARGET_VARIABLE)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(plt)

                else:
                    st.warning(f"Target variable '{TARGET_VARIABLE}' or required features not found after processing.")

            else:
                st.warning("Please select a valid Target Variable and at least one Market Feature present in the data.")

        else:
            st.warning("Please select Target Variable, Market Feature(s), and forecast period.")

    except ValueError as ve:
        st.error(f"Error in data processing: {ve}")
    except Exception as e:
        st.error(f"❌ Error loading prediction data: {e}")
else:
    st.info("Click on 'LNG Market' or 'LNG Market Prediction' in the sidebar to view the respective dashboards.")