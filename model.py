import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load training data
try:
    df = pd.read_csv('train_data.csv')
    print(f"Training data loaded: {len(df)} rows")
except:
    print("Error: train_data.csv not found. Run process_data.py first.")
    exit()

# Define features and target
features = [
    'age_months', 'sex', 'birth_weight_kg', 'feeding_practice',
    'gestational_age_weeks', 'delivery_mode', 'maternal_age',
    'weight_for_age_zscore', 'height_for_age_zscore', 'weight_for_height_zscore'
]
target = 'Risk_Level'

X = df[features]
y = df[target]

# Define preprocessing for numeric and categorical columns
numeric_features = [
    'age_months', 'birth_weight_kg', 'gestational_age_weeks', 'maternal_age',
    'weight_for_age_zscore', 'height_for_age_zscore', 'weight_for_height_zscore'
]
categorical_features = ['sex', 'feeding_practice', 'delivery_mode']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Initialize Model
# Using class_weight='balanced' to handle potential data imbalance
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Create pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

# Hyperparameter tuning
# Limited max_depth to prevent overfitting on the z-score noise
param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [10, 12, 15],
    'classifier__min_samples_split': [5, 10]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Starting Grid Search to find best parameters...")
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

# Output results
print("\n" + "-"*30)
print(f"Best Validation Accuracy: {grid_search.best_score_*100:.2f}%")
print(f"Best Params: {grid_search.best_params_}")
print("-"*30)

# Save the best model
joblib.dump(grid_search.best_estimator_, 'nutrition_model.pkl')
print("Model saved to nutrition_model.pkl")