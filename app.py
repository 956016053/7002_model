import gradio as gr
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained model
print("Loading model...")
try:
    model = joblib.load('nutrition_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'nutrition_model.pkl' is in the same folder.")


# 2. Define prediction function
def predict_risk(age, sex, birth_weight, feeding, gest_weeks, delivery, mat_age,
                 w_age_z, h_age_z, w_height_z):
    # Prepare input dataframe
    # The columns MUST match the training data exactly
    input_df = pd.DataFrame({
        'age_months': [age],
        'sex': [sex],
        'birth_weight_kg': [birth_weight],
        'feeding_practice': [feeding],
        'gestational_age_weeks': [gest_weeks],
        'delivery_mode': [delivery],
        'maternal_age': [mat_age],
        # Important: The model expects these Z-scores now
        'weight_for_age_zscore': [w_age_z],
        'height_for_age_zscore': [h_age_z],
        'weight_for_height_zscore': [w_height_z]
    })

    # Make prediction
    try:
        # Get probability for smoother output
        proba = model.predict_proba(input_df)[0]  # [prob_low, prob_high]
        classes = model.classes_

        # Find index of 'High'
        high_index = list(classes).index('High')
        risk_score = proba[high_index]

        # Threshold logic (matches our optimization)
        if risk_score > 0.5:
            return (
                f"🔴 **HIGH RISK DETECTED**\n\n"
                f"Probability: {risk_score * 100:.1f}%\n"
                f"Recommendation: Immediate clinical assessment required."
            )
        else:
            return (
                f"🟢 **Low Risk**\n\n"
                f"Probability of Risk: {risk_score * 100:.1f}%\n"
                f"Child appears to be developing within normal range."
            )

    except Exception as e:
        return f"Error during prediction: {str(e)}"


# 3. Create Gradio Interface
# Inputs based on our 10-feature model
inputs = [
    # --- Basic Demographics ---
    gr.Slider(0, 60, step=1, label="Child Age (Months)"),
    gr.Radio(["Male", "Female"], label="Sex"),
    gr.Number(label="Birth Weight (kg)", value=3.0),
    gr.Dropdown(["Exclusive breastfeeding", "Formula feeding", "Mixed feeding", "Breastfeeding"],
                label="Feeding Practice"),

    # --- Maternal & Birth History ---
    gr.Number(label="Gestational Age (Weeks)", value=39),
    gr.Radio(["Vaginal", "C-section"], label="Delivery Mode"),
    gr.Number(label="Maternal Age (Years)", value=25),

    # --- Clinical Indicators (Z-Scores) ---
    # These are the "Cheat Codes" for the model.
    # Entering -3 here will almost guarantee a High Risk result.
    gr.Number(label="Weight-for-Age Z-score (Standard: 0)", value=0),
    gr.Number(label="Height-for-Age Z-score (Standard: 0)", value=0),
    gr.Number(label="Weight-for-Height Z-score (Standard: 0)", value=0)
]

outputs = gr.Markdown(label="Prediction Result")

app = gr.Interface(
    fn=predict_risk,
    inputs=inputs,
    outputs=outputs,
    title="Child Nutrition Risk Diagnostic Tool",
    description="""
    This tool uses a Random Forest model to assess malnutrition risk in children under 5 years.
    It integrates **WHO Growth Standards (Z-scores)** for diagnostic precision.
    """,
    theme="default"
)

# 4. Launch
if __name__ == "__main__":
    # strictly for Hugging Face Spaces
    app.launch()