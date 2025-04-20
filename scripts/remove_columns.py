import pandas as pd

# Set display options for better output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# --- 1. Load the Dataset ---
print("Loading data from CharlesBookClub.csv...")
# Read the CSV file into a DataFrame, specifying the index column
df = pd.read_csv('CharlesBookClub.csv', index_col=0)
print("Data loaded successfully.")
print("Initial columns:", df.columns.tolist())

# --- 2. Create Binary Target Column ---
print("\nCreating 'Florence_Buyer' target column...")
# Create the binary target column `Florence_Buyer`
# If `Yes_Florence` is 1, `Florence_Buyer` is 1, otherwise 0
df['Florence_Buyer'] = df['Yes_Florence'].apply(lambda x: 1 if x == 1 else 0)
print("'Florence_Buyer' column created.")
# Display value counts for the new column
print(df['Florence_Buyer'].value_counts())


# --- 3. Remove Irrelevant/Redundant Columns ---
print("\nRemoving irrelevant/redundant columns...")
# Define columns to remove
cols_to_remove = ['ID#', 'Mcode', 'Rcode', 'Fcode', 'Yes_Florence', 'No_Florence', 'Florence']
# Check which columns exist before attempting removal
existing_cols_to_remove = [col for col in cols_to_remove if col in df.columns]
df.drop(columns=existing_cols_to_remove, inplace=True)
print(f"Removed columns: {existing_cols_to_remove}")
print("Remaining columns:", df.columns.tolist())


# --- 4. Check for Missing Values ---
print("\nChecking for missing values...")
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0]) # Only print columns with missing values
if missing_values.sum() == 0:
    print("No missing values found.")


# --- 5. Display Final DataFrame Info ---
print("\nFinal DataFrame structure:")
# Display the first 5 rows of the cleaned DataFrame
print("First 5 rows:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Print the column names and their data types
print("\nColumn names and data types:")
print(df.info())

print("\nData preparation complete.")