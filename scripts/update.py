import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Ignore warnings related to convergence or model fitting issues due to small data
warnings.filterwarnings("ignore")

# --- Preprocessing Code ---
# Ensure the file 'bicup2006.csv' is accessible
df = pd.read_csv('bicup2006.csv')
df['DATE_TIME'] = df['DATE'] + ' ' + df['TIME']
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%d-%b-%y %H:%M')
df_agg = df.groupby(df['DATE_TIME'].dt.date).agg(
    DEMAND=('DEMAND', 'sum'),
    TIME=('DATE_TIME', 'min')
).reset_index()
df_agg = df_agg.rename(columns={'DATE_TIME': 'DATE'})
df_agg['DATETIME'] = pd.to_datetime(df_agg['DATE'].astype(str) + ' ' + df_agg['TIME'].dt.strftime('%H:%M'))
# Ensure the index is a DatetimeIndex and set frequency if possible (Daily)
df_time_series = df_agg.set_index('DATETIME').drop(columns=['DATE', 'TIME']).asfreq('D')

# --- Train/Test Split ---
# WARNING: Splitting 21 points is not statistically reliable. This is for demonstration only.
n_total = len(df_time_series)
n_train = 17 # Approximately 80% for training
n_test = n_total - n_train # Remaining 4 points for testing

metrics_summary_df = pd.DataFrame() # Initialize empty DataFrame

# Check if split is minimally feasible
if n_train < 14 or n_test < 1:
     print(f"ERROR: Not enough data for a meaningful train/test split (Train: {n_train}, Test: {n_test}). Cannot proceed with evaluation.")
else:
    train_data = df_time_series.iloc[:n_train]
    test_data = df_time_series.iloc[n_train:]
    print(f"--- Train/Test Split ---")
    print(f"Train set size: {len(train_data)} points")
    print(f"Test set size: {len(test_data)} points")
    print("-" * 25)

    # --- Model Training & Forecasting on Test Set Period ---
    forecasts = {}

    # 1. SARIMA
    sarima_order = (1, 1, 0)
    sarima_seasonal_order = (1, 1, 0, 7)
    print(f"\nAttempting SARIMA{sarima_order}x{sarima_seasonal_order} on training data...")
    try:
        sarima_model_train = SARIMAX(train_data['DEMAND'], order=sarima_order, seasonal_order=sarima_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        sarima_results_train = sarima_model_train.fit(disp=False)
        forecast_values = sarima_results_train.forecast(steps=n_test)
        forecasts['SARIMA'] = forecast_values
        print("SARIMA fitting and forecasting successful.")
    except Exception as e:
        print(f"SARIMA failed: {e}. Assigning NaN forecast.")
        forecasts['SARIMA'] = pd.Series(np.nan, index=test_data.index) # Ensure index matches test set

    # 2. Holt-Winters Exponential Smoothing
    print("\nAttempting Holt-Winters Exponential Smoothing on training data...")
    try:
        # Using simple heuristic initialization for stability on small data
        hw_model_train = ExponentialSmoothing(train_data['DEMAND'], trend='add', seasonal='add', seasonal_periods=7, initialization_method='heuristic')
        hw_results_train = hw_model_train.fit()
        forecast_values = hw_results_train.forecast(steps=n_test)
        forecasts['HoltWinters'] = forecast_values
        print("Holt-Winters fitting and forecasting successful.")
    except Exception as e:
        print(f"Holt-Winters failed: {e}. Assigning NaN forecast.")
        forecasts['HoltWinters'] = pd.Series(np.nan, index=test_data.index)

    # 3. Seasonal Naive Method
    print("\nGenerating Seasonal Naive forecast based on training data...")
    if len(train_data) >= 7:
        last_train_week_demand = train_data['DEMAND'][-7:]
        naive_forecast_list = list(last_train_week_demand) * (n_test // 7 + (n_test % 7 > 0))
        forecast_values = pd.Series(naive_forecast_list[:n_test], index=test_data.index)
        forecasts['SeasonalNaive'] = forecast_values
        print("Seasonal Naive forecast generated.")
    else:
        print("Not enough training data for Seasonal Naive (need >= 7). Assigning NaN forecast.")
        forecasts['SeasonalNaive'] = pd.Series(np.nan, index=test_data.index)


    # --- Evaluate Forecasts on Test Set ---
    print("\n--- Evaluation Metrics on Test Set ---")
    actual_values = test_data['DEMAND']
    results_list = []

    for model_name, forecast_values in forecasts.items():
        model_metrics = {"Model": model_name}
        # Check if forecast is all NaN (model failed)
        if forecast_values.isnull().all():
            print(f"{model_name}: Model failed or produced NaN forecast.")
            model_metrics.update({"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan})
        else:
            # Ensure indices align for calculation
            aligned_actual, aligned_forecast = actual_values.align(forecast_values, join='inner')
            if not aligned_forecast.empty:
                 mae = mean_absolute_error(aligned_actual, aligned_forecast)
                 mse = mean_squared_error(aligned_actual, aligned_forecast)
                 rmse = np.sqrt(mse)
                 print(f"{model_name} -> MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")
                 model_metrics.update({"MAE": mae, "MSE": mse, "RMSE": rmse})
            else:
                 print(f"{model_name}: Forecast alignment failed.")
                 model_metrics.update({"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan})
        results_list.append(model_metrics)

    metrics_summary_df = pd.DataFrame(results_list) # Assign results here
    print("\n--- Summary Table ---")
    print(metrics_summary_df.to_markdown(index=False, floatfmt=".2f"))

    print("\nWARNING: Evaluation based on only 4 test points is highly unreliable.")
    print("-" * 25)

# Final check using the corrected DataFrame method '.empty'
if metrics_summary_df.empty:
     print("\nNo evaluation metrics generated, likely due to insufficient data for train/test split.")