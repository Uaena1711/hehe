import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
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

# Check if split is minimally feasible (need >= 2*seasonal_period for HW/SARIMA)
if n_train < 14 or n_test < 1:
     print(f"ERROR: Not enough data for a meaningful train/test split (Train: {n_train}, Test: {n_test}). Cannot proceed.")
     # Exit or handle error appropriately in a real scenario
     # For demonstration, we'll let it continue but results are invalid.
     train_data = df_time_series.iloc[:n_train]
     test_data = df_time_series.iloc[n_train:] # Might be empty if n_test < 1
else:
    train_data = df_time_series.iloc[:n_train]
    test_data = df_time_series.iloc[n_train:]
    print(f"--- Train/Test Split ---")
    print(f"Train set size: {len(train_data)} points (ends {train_data.index[-1]})")
    print(f"Test set size: {len(test_data)} points (starts {test_data.index[0]})")
    print("-" * 25)

# --- Model Training & Forecasting on Test Set Period ---
# Dictionary to store forecasts
forecasts = {}
evaluation_results = {}

# 1. SARIMA
# WARNING: Fitting SARIMA on 17 points is highly likely to fail or be unstable.
sarima_order = (1, 1, 0)
sarima_seasonal_order = (1, 1, 0, 7)
print(f"\nAttempting SARIMA{sarima_order}x{sarima_seasonal_order} on training data...")
try:
    sarima_model_train = SARIMAX(train_data['DEMAND'], order=sarima_order, seasonal_order=sarima_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    sarima_results_train = sarima_model_train.fit(disp=False)
    # Forecast for the test period length
    forecast_values = sarima_results_train.forecast(steps=n_test)
    forecasts['SARIMA'] = forecast_values
    print("SARIMA fitting and forecasting successful.")
except Exception as e:
    print(f"SARIMA failed: {e}. Assigning NaN forecast.")
    forecasts['SARIMA'] = pd.Series(np.nan, index=test_data.index) # Ensure index matches test set

# 2. Holt-Winters Exponential Smoothing
# WARNING: Fitting Holt-Winters on 17 points provides unreliable parameters.
print("\nAttempting Holt-Winters Exponential Smoothing on training data...")
try:
    hw_model_train = ExponentialSmoothing(train_data['DEMAND'], trend='add', seasonal='add', seasonal_periods=7)
    hw_results_train = hw_model_train.fit()
    # Forecast for the test period length
    forecast_values = hw_results_train.forecast(steps=n_test)
    forecasts['HoltWinters'] = forecast_values
    print("Holt-Winters fitting and forecasting successful.")
except Exception as e:
    print(f"Holt-Winters failed: {e}. Assigning NaN forecast.")
    forecasts['HoltWinters'] = pd.Series(np.nan, index=test_data.index)

# 3. Seasonal Naive Method
# Base the naive forecast on the last season *within the training data*
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
print("\n--- Evaluation on Test Set ---")
# Ensure test_data is not empty before proceeding
if not test_data.empty:
    actual_values = test_data['DEMAND']
    for model_name, forecast_values in forecasts.items():
        # Check if forecast is all NaN (model failed)
        if forecast_values.isnull().all():
            print(f"{model_name}: Model failed, no evaluation possible.")
            evaluation_results[model_name] = np.nan
        else:
            # Ensure indices align for calculation (should match if code above is correct)
            aligned_actual, aligned_forecast = actual_values.align(forecast_values, join='inner')
            if not aligned_forecast.empty:
                 mae = mean_absolute_error(aligned_actual, aligned_forecast)
                 print(f"{model_name} MAE: {mae:.2f}")
                 evaluation_results[model_name] = mae
            else:
                 print(f"{model_name}: Forecast alignment failed, no evaluation possible.")
                 evaluation_results[model_name] = np.nan

    print("WARNING: Evaluation based on only 4 test points is highly unreliable.")
else:
    print("Test set is empty or too small, skipping evaluation.")
print("-" * 25)

# --- Plotting Results ---
print("\nPlotting train/test forecasts...")
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")
plt.plot(train_data.index, train_data['DEMAND'], label='Training Data', marker='.')
if not test_data.empty: # Only plot test data if it exists
    plt.plot(test_data.index, test_data['DEMAND'], label='Actual Test Data', marker='o', color='black')

# Plot forecasts if they exist
for model_name, forecast_values in forecasts.items():
     if not forecast_values.isnull().all(): # Only plot if model didn't fail
        plt.plot(forecast_values.index, forecast_values, label=f'{model_name} Forecast', marker='x', linestyle='--')

plt.title('Forecast vs Actual on Test Set (DEMO ONLY - Unreliable due to small data)')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()