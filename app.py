import gradio as gr
import joblib
import pandas as pd

# load model
try:
    model = joblib.load('nutrition_model.pkl')
    print("model loaded.")
except:
    print("error: run model.py first")

# options (hardcoded from common data)
feeding_options = ["Exclusive breastfeeding", "Formula feeding", "Mixed feeding", "Breastfeeding"]
delivery_options = ["Vaginal", "C-section"] # Based on your dataset screenshot

def predict_risk(age, sex, weight, feeding, gest_weeks, delivery, mat_age):
    # DataFrame columns must match training data EXACTLY
    input_data = pd.DataFrame({
        'age_months': [age],
        'sex': [sex],
        'birth_weight_kg': [weight],
        'feeding_practice': [feeding],
        'gestational_age_weeks': [gest_weeks],
        'delivery_mode': [delivery],
        'maternal_age': [mat_age]
    })
    
    pred = model.predict(input_data)[0]
    
    if pred == 'Low':
        return "🟢 **Low Risk**\nChild is developing well."
    else:
        return "🔴 **High Risk**\nRisk factors detected (Stunting/Wasting)."

# Expanded Interface
iface = gr.Interface(
    fn=predict_risk,
    inputs=[
        # Basic Info
        gr.Slider(0, 60, step=1, label="Child Age (Months)"),
        gr.Radio(["Male", "Female"], label="Sex"),
        gr.Number(label="Birth Weight (kg)", value=3.0),
        gr.Dropdown(feeding_options, label="Feeding Practice"),
        
        # New Medical Indicators
        gr.Number(label="Gestational Age (Weeks)", value=39), # 孕周
        gr.Radio(delivery_options, label="Delivery Mode"),    # 分娩方式
        gr.Number(label="Maternal Age (Years)", value=25)     # 母亲年龄
    ],
    outputs="markdown",
    title="Advanced Nutrition Predictor (7002 Assignment)",
    description="Now includes maternal and birth history for better accuracy."
)

print("starting server on port 6006...")
iface.launch(server_name="0.0.0.0", server_port=6006)