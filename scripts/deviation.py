import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn is often used for styling, though not strictly necessary for the plot itself

# --- Preprocessing Code (Needed to create df_time_series) ---
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

# --- Rolling Mean & Standard Deviation Plot Visualization Code ---

print("\nCalculating and Plotting Rolling Statistics...")
# Define window size (e.g., 7 for weekly pattern)
window_size = 7

# Calculate rolling mean and std deviation
rolling_mean = df_time_series['DEMAND'].rolling(window=window_size).mean()
rolling_std = df_time_series['DEMAND'].rolling(window=window_size).std()

# Set plot style (optional)
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df_time_series['DEMAND'], color='blue', label='Original Demand')
plt.plot(rolling_mean, color='red', label=f'Rolling Mean (window={window_size})')
plt.plot(rolling_std, color='black', label=f'Rolling Std Dev (window={window_size})')

# Add titles and labels
plt.title('Rolling Mean & Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend(loc='best')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Ensure layout is neat
plt.tight_layout()

# Show the plot
plt.show()

print("Plotting code is ready.")