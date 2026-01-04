import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load dataset
# Try loading the augmented one first, else load the original
try:
    df = pd.read_csv('child_nutrition_dataset_augmented.csv', low_memory=False, on_bad_lines='skip')
    print("Loaded augmented dataset.")
except:
    df = pd.read_csv('child_nutrition_dataset.csv')
    print("Loaded original dataset.")

# 2. Select features
# We include z-scores because they are critical for diagnosis
numeric_cols = [
    'age_months', 'birth_weight_kg', 'gestational_age_weeks', 'maternal_age',
    'weight_for_age_zscore', 'height_for_age_zscore', 'weight_for_height_zscore'
]

# Convert columns to numeric, force errors to NaN then fill with 0
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# === Add noise to Z-scores ===
# This simulates real-world measurement errors and prevents the model
# from simply memorizing the rules (overfitting), making it more robust.
np.random.seed(42)
noise_level = 0.35

print(f"Adding noise to z-scores (level={noise_level})...")
z_score_cols = ['weight_for_age_zscore', 'height_for_age_zscore', 'weight_for_height_zscore']

for col in z_score_cols:
    if col in df.columns:
        # Add random noise
        noise = np.random.normal(0, noise_level, size=len(df))
        df[col] = df[col] + noise


# 3. Create target label 'Risk_Level'
# Logic: If stunted or wasted, risk is High. Otherwise Low.
def get_risk_label(row):
    s = str(row.get('stunted', '')).upper()
    w = str(row.get('wasted', '')).upper()

    # Check for various true values (True, 'TRUE', 1, '1')
    is_high = (s == 'TRUE' or s == '1') or (w == 'TRUE' or w == '1')
    return 'High' if is_high else 'Low'


df['Risk_Level'] = df.apply(get_risk_label, axis=1)

# 4. Final feature selection
features = [
    'age_months', 'sex', 'birth_weight_kg', 'feeding_practice',
    'gestational_age_weeks', 'delivery_mode', 'maternal_age',
    'weight_for_age_zscore', 'height_for_age_zscore', 'weight_for_height_zscore'
]

# Fill categorical missing values
cat_cols = ['sex', 'feeding_practice', 'delivery_mode']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str)

# 5. Split data
X = df[features].fillna(0)
y = df['Risk_Level']

# Stratified split to keep class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save processed files
train_output = pd.concat([X_train, y_train], axis=1)
test_output = pd.concat([X_test, y_test], axis=1)

train_output.to_csv('train_data.csv', index=False)
test_output.to_csv('test_data.csv', index=False)

print("Data processing done. Files saved.")