import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("Loading model and test data...")

    # Load model
    try:
        model = joblib.load('nutrition_model.pkl')
    except:
        print("Model file not found.")
        return

    # Load test data
    try:
        df_test = pd.read_csv('test_data.csv')
    except:
        print("Test data file not found.")
        return

    # Feature list (must match training)
    features = [
        'age_months', 'sex', 'birth_weight_kg', 'feeding_practice',
        'gestational_age_weeks', 'delivery_mode', 'maternal_age',
        'weight_for_age_zscore', 'height_for_age_zscore', 'weight_for_height_zscore'
    ]
    target = 'Risk_Level'

    # Check columns
    if not all(col in df_test.columns for col in features):
        print("Error: Missing columns in test data.")
        return

    X_test = df_test[features]
    y_true = df_test[target]

    print(f"Validating on {len(df_test)} samples...")

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    print("\n=== Validation Report ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()