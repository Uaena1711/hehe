import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
FILE_PATH = 'CharlesBookClub.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
ADJUSTED_THRESHOLD = 0.2 # Threshold for classifying as 'Buyer'

# --- 1. Load Data ---
print(f"Loading data from {FILE_PATH}...")
try:
    df = pd.read_csv(FILE_PATH, index_col=0)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit() # Exit if file not found

# --- 2. Preprocessing ---
print("\nPreprocessing data...")
# Create binary target column 'Florence_Buyer'
df['Florence_Buyer'] = df['Yes_Florence'].apply(lambda x: 1 if x == 1 else 0)
print("'Florence_Buyer' column created.")
print("Value Counts:\n", df['Florence_Buyer'].value_counts())

# Define and remove irrelevant/redundant columns
cols_to_remove = ['ID#', 'Mcode', 'Rcode', 'Fcode', 'Yes_Florence', 'No_Florence', 'Florence']
existing_cols_to_remove = [col for col in cols_to_remove if col in df.columns]
df.drop(columns=existing_cols_to_remove, inplace=True)
print(f"Removed columns: {existing_cols_to_remove}")

# Check for missing values
if df.isnull().sum().sum() == 0:
    print("No missing values found.")
else:
    print("Warning: Missing values found. Handling may be required.")
    print(df.isnull().sum())

# --- 3. Define Features (X) and Target (y) ---
X = df.drop('Florence_Buyer', axis=1)
y = df['Florence_Buyer']
print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- 4. Split Data ---
print(f"\nSplitting data into {1-TEST_SIZE:.0%} training and {TEST_SIZE:.0%} testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Data split complete.")
print("Training set distribution:\n", y_train.value_counts(normalize=True))
print("Testing set distribution:\n", y_test.value_counts(normalize=True))

# --- 5. Train Random Forest Model ---
print("\nTraining Random Forest Classifier...")
# Using class_weight='balanced' to help with imbalance
rf_classifier = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_estimators=100 # Default number of trees
)
rf_classifier.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Evaluate with Default Threshold (0.5) ---
print("\n--- Evaluating Model (Default Threshold = 0.5) ---")
y_pred_default = rf_classifier.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)
cm_default = confusion_matrix(y_test, y_pred_default)

print(f"Accuracy: {accuracy_default:.4f}")
print("\nConfusion Matrix:")
print(cm_default)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default, target_names=['Non-Buyer (0)', 'Buyer (1)']))

# --- 7. Evaluate with Adjusted Threshold ---
print(f"\n--- Evaluating Model (Adjusted Threshold = {ADJUSTED_THRESHOLD}) ---")
# Get predicted probabilities for the positive class ('Buyer')
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Apply the adjusted threshold
y_pred_adjusted = (y_pred_proba >= ADJUSTED_THRESHOLD).astype(int)

accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)

print(f"Accuracy: {accuracy_adjusted:.4f}")
print("\nConfusion Matrix:")
print(cm_adjusted)

# Plot Confusion Matrix for Adjusted Threshold
plt.figure(figsize=(6, 4))
sns.heatmap(cm_adjusted, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Non-Buyer', 'Predicted Buyer'], yticklabels=['Actual Non-Buyer', 'Actual Buyer'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Confusion Matrix (Threshold = {ADJUSTED_THRESHOLD})')
plt.show()


print("\nClassification Report:")
print(classification_report(y_test, y_pred_adjusted, target_names=['Non-Buyer (0)', 'Buyer (1)']))

# --- 8. Feature Importance ---
print("\n--- Feature Importance Analysis ---")
importances = rf_classifier.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df.round(4).to_markdown(index=False, numalign="left", stralign="left"))

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance - Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\nScript finished.")