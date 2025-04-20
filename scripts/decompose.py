import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Optional for styling
from statsmodels.tsa.seasonal import seasonal_decompose

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

# --- Seasonal Decomposition Code ---

print("\nPerforming Seasonal Decomposition...")

# Define the seasonal period (7 for weekly data)
seasonal_period = 7
# Choose model type ('additive' or 'multiplicative')
# Additive was used previously and seemed appropriate.
model_type = 'additive'

try:
    # Perform decomposition
    # Ensure the series has enough data (at least 2 * seasonal_period)
    decomposition = seasonal_decompose(
        df_time_series['DEMAND'],
        model=model_type,
        period=seasonal_period
    )

    # Plot decomposition results
    print(f"Plotting decomposition results (Model={model_type}, Period={seasonal_period})...")
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.suptitle(f'Seasonal Decomposition ({model_type.capitalize()}, Period={seasonal_period})', y=1.0) # Add overall title
    plt.show()


    print("\nDecomposition successful and plot prepared.")

except ValueError as e:
    print(f"\nDecomposition failed: {e}")
    print("This often happens if the time series is too short for the specified period.")