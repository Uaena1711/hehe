import pandas as pd
# You need to install mlxtend: pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# --- Configuration ---
FILE_PATH = 'CharlesBookClub.csv' # Make sure this file is in the same directory or provide the full path
MIN_SUPPORT = 0.05  # Minimum support threshold (e.g., 5%)
MIN_CONFIDENCE = 0.5 # Minimum confidence threshold (e.g., 50%)
MIN_LIFT = 1.0       # Minimum lift threshold (rules with lift <= 1 are generally not interesting)

# --- 1. Load Data ---
print(f"Loading data from {FILE_PATH}...")
try:
    # Read the original CSV, assuming the first column is the index
    df = pd.read_csv(FILE_PATH, index_col=0)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

# --- 2. Prepare Data for ARM ---
# Define the book genre columns
book_genre_cols = ['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks',
                   'RefBks', 'ArtBks', 'GeogBks', 'ItalCook',
                   'ItalAtlas', 'ItalArt']

# Verify columns exist in df
missing_cols = [col for col in book_genre_cols if col not in df.columns]
if missing_cols:
     print(f"Error: The following genre columns are missing from the DataFrame: {missing_cols}")
     # Decide how to handle: exit or proceed with available columns
     available_genre_cols = [col for col in book_genre_cols if col in df.columns]
     if not available_genre_cols:
         print("Error: No suitable book genre columns found. Exiting.")
         exit()
     else:
         print(f"Warning: Proceeding with available columns: {available_genre_cols}")
         book_genre_cols = available_genre_cols # Use only available columns
else:
    print("All expected genre columns found.")

# Select genre columns
genre_df = df[book_genre_cols]

# Binarize the data (1 if purchased > 0, else 0)
print("Binarizing purchase data (1 if bought, 0 otherwise)...")
genre_df_binary = genre_df.applymap(lambda x: 1 if x > 0 else 0)
print("Binarized data preview:")
print(genre_df_binary.head())

# --- 3. Apply Apriori Algorithm ---
print(f"\nApplying Apriori algorithm with min_support = {MIN_SUPPORT}...")
try:
    frequent_itemsets = apriori(genre_df_binary,
                                min_support=MIN_SUPPORT,
                                use_colnames=True)
except Exception as e:
    print(f"An error occurred during Apriori execution: {e}")
    exit()

if frequent_itemsets.empty:
    print("No frequent itemsets found with the specified minimum support.")
    print("Consider lowering the MIN_SUPPORT value.")
    exit()
else:
    print(f"\nFound {len(frequent_itemsets)} frequent itemsets.")
    print("Top 10 frequent itemsets by support:")
    print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

# --- 4. Generate Association Rules ---
print(f"\nGenerating association rules with min_confidence = {MIN_CONFIDENCE}...")
try:
    rules = association_rules(frequent_itemsets,
                              metric="confidence",
                              min_threshold=MIN_CONFIDENCE)
except Exception as e:
    print(f"An error occurred during rule generation: {e}")
    exit()

if rules.empty:
    print("No association rules generated with the specified minimum confidence.")
    print("Consider lowering the MIN_CONFIDENCE value or checking frequent itemsets.")
    exit()
else:
     print(f"\nGenerated {len(rules)} rules initially.")
a
# --- 5. Filter and Sort Rules ---
print(f"\nFiltering rules with lift >= {MIN_LIFT}...")
rules_filtered = rules[rules['lift'] >= MIN_LIFT]

if rules_filtered.empty:
    print("No rules found meeting the minimum lift criterion.")
    exit()
else:
    print(f"Found {len(rules_filtered)} rules meeting lift criterion.")
    # Sort rules by lift (you can change to 'confidence' or 'support' if preferred)
    rules_sorted = rules_filtered.sort_values(by='lift', ascending=False)
    print("\nTop Association Rules (sorted by Lift):")
    # Displaying results with rounded values for clarity
    print(rules_sorted.round({'support': 3, 'confidence': 3, 'lift': 3, 'leverage': 3, 'conviction': 3}).to_markdown(index=False))

    # --- 6. Interpretation Help (Top 3 Rules) ---
    print("\n--- Interpretation of Top 3 Rules (by Lift) ---")
    top_3_rules = rules_sorted.head(3)

    for i, idx in enumerate(top_3_rules.index):
        rule = top_3_rules.loc[idx]
        antecedent = ", ".join(list(rule['antecedents']))
        consequent = ", ".join(list(rule['consequents']))
        support = rule['support']
        confidence = rule['confidence']
        lift = rule['lift']

        print(f"\nRule {i+1}: IF customer buys ({antecedent}) THEN they also buy ({consequent})")
        print(f"   - Support: {support:.3f} ({support*100:.1f}% of all transactions contain both)")
        print(f"   - Confidence: {confidence:.3f} ({confidence*100:.1f}% of customers buying '{antecedent}' also buy '{consequent}')")
        print(f"   - Lift: {lift:.3f} (Purchasing '{consequent}' is {lift:.1f} times more likely if '{antecedent}' is purchased, compared to average)")
        print(f"   - Actionable Insight Hint: Lift > 1 indicates a positive association. Higher lift suggests stronger association. Consider bundling, recommendations, or co-placement.")

print("\nScript finished.")