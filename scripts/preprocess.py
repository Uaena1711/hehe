import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Preprocessing Code ---
# Read the CSV file into a DataFrame
# Ensure the file 'bicup2006.csv' is accessible
df = pd.read_csv('bicup2006.csv')

# Combine 'DATE' and 'TIME' columns into a single 'DATE_TIME' column
df['DATE_TIME'] = df['DATE'] + ' ' + df['TIME']

# Convert 'DATE_TIME' column to datetime objects, specifying the format
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%d-%b-%y %H:%M')

# Group by date and aggregate: sum 'DEMAND' and find the minimum 'TIME'
df_agg = df.groupby(df['DATE_TIME'].dt.date).agg(
    DEMAND=('DEMAND', 'sum'),
    TIME=('DATE_TIME', 'min') # Keep the earliest time for each date
).reset_index()

# Rename the date column back to 'DATE'
df_agg = df_agg.rename(columns={'DATE_TIME': 'DATE'})

# Combine the aggregated date and the earliest time back into a single datetime column
df_agg['DATETIME'] = pd.to_datetime(df_agg['DATE'].astype(str) + ' ' + df_agg['TIME'].dt.strftime('%H:%M'))

# Set 'DATETIME' as the index
df_time_series = df_agg.set_index('DATETIME')

# Drop the original 'DATE' and 'TIME' columns
df_time_series = df_time_series.drop(columns=['DATE', 'TIME'])


# --- Forecasting Code ---
# Set the plot style
sns.set_style("whitegrid")

# Get the last 7 days of data to use as the seasonal pattern
last_week_demand = df_time_series['DEMAND'][-7:]

# Generate the dates for the next 14 days
forecast_dates = pd.date_range(start=df_time_series.index[-1] + pd.Timedelta(days=1), periods=14, freq='D')

# Create the forecast by repeating the last week's pattern twice
forecast_values = list(last_week_demand) * 2

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({'DATE': forecast_dates, 'FORECAST_DEMAND': forecast_values})

# Set Date as index
forecast_df = forecast_df.set_index('DATE')

# --- Plotting Code ---
# (This part generates the plot visualization)
plt.figure(figsize=(12, 6))
plt.plot(df_time_series.index, df_time_series['DEMAND'], marker='o', linestyle='-', label='Historical Demand')
plt.plot(forecast_df.index, forecast_df['FORECAST_DEMAND'], marker='x', linestyle='--', label='Forecasted Demand')

# Add titles and labels
plt.title('Demand Forecast vs Historical Data')
plt.xlabel('Date')
plt.ylabel('Demand (Units Sold)')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Prepare the plot to be shown
plt.tight_layout()
plt.show()