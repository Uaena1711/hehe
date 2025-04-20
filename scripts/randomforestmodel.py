import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check if the model 'rf_classifier' exists from previous steps
try:
    # Get feature importances from the trained model
    importances = rf_classifier.feature_importances_
    # Get feature names
    feature_names = X_train.columns # Assuming X_train holds the feature columns used for training

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("Feature Importances from Random Forest Model:")
    print(feature_importance_df.round(4).to_markdown(index=False, numalign="left", stralign="left"))

    # --- Plotting the Feature Importances ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance - Random Forest Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

except NameError:
    print("Error: The Random Forest model ('rf_classifier') or training data ('X_train') was not found.")
    print("Please ensure the model training code was executed successfully earlier.")
except Exception as e:
    print(f"An error occurred: {e}")