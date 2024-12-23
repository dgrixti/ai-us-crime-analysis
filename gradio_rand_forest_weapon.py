import gradio as gr
import pandas as pd
import joblib
import random

# Load the saved model and preprocessor
saved_objects = joblib.load('random_forest_weapon_model.pkl')  # Updated model file
model = saved_objects  # Directly loaded model
preprocessor = joblib.load('weapon_category_model.pkl')['preprocessor']  # Preprocessor from earlier saved objects

# Input feature names (relevant for the Weapon Category prediction)
input_features = [
    'Region', 'Season', 'Victim Sex', 'Victim Race', 'Victim Ethnicity',
    'Perpetrator Sex', 'Perpetrator Race', 'Perpetrator Ethnicity',
    'Agency Type', 'Agency_Type_grouped', 'Relationship Category', 'Victim Age'
]

# Options for dropdowns
region_options = [
    'West Coast', 'South', 'Mountain West', 'East Coast', 'Midwest', 'Pacific Northwest'
]
season_options = ['Winter', 'Spring', 'Summer', 'Autumn']
sex_options = ['Male', 'Female']
race_options = ['White', 'Black', 'Asian/Pacific Islander', 'Native American/Alaska Native']
ethnicity_options = ['Hispanic', 'Not Hispanic']
agency_type_options = ['Municipal Police', 'Sheriff', 'Other Police']
relationship_category_options = ['Family', 'Lover', 'Stranger', 'Acquaintance']

# Random value generator
def get_random_value(options):
    return random.choice(options)

# Prediction function for Weapon Category
def predict_weapon_category(region, season, victim_sex, victim_race, victim_ethnicity,
                            perpetrator_sex, perpetrator_race, perpetrator_ethnicity,
                            agency_type, relationship_category, victim_age):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Region': region,
        'Season': season,
        'Victim Sex': victim_sex,
        'Victim Race': victim_race,
        'Victim Ethnicity': victim_ethnicity,
        'Perpetrator Sex': perpetrator_sex,
        'Perpetrator Race': perpetrator_race,
        'Perpetrator Ethnicity': perpetrator_ethnicity,
        'Agency Type': agency_type,
        'Relationship Category': relationship_category,
        'Victim Age': victim_age
    }])

    # Assign Agency_Type_grouped to be the same as Agency Type
    input_data['Agency_Type_grouped'] = input_data['Agency Type']

    # Ensure input columns match preprocessor expectations
    input_data = input_data[input_features]

    # Preprocess input data
    input_transformed = preprocessor.transform(input_data)

    # Predict probabilities
    probabilities = model.predict_proba(input_transformed)[0]
    class_labels = model.classes_

    # Return predictions as a dictionary with float values
    return {label: prob for label, prob in zip(class_labels, probabilities)}


# Gradio interface
interface = gr.Interface(
    fn=predict_weapon_category,
    inputs=[
        gr.Dropdown(region_options, label="Region", value=get_random_value(region_options)),
        gr.Dropdown(season_options, label="Season", value=get_random_value(season_options)),
        gr.Dropdown(sex_options, label="Victim Sex", value=get_random_value(sex_options)),
        gr.Dropdown(race_options, label="Victim Race", value=get_random_value(race_options)),
        gr.Dropdown(ethnicity_options, label="Victim Ethnicity", value=get_random_value(ethnicity_options)),
        gr.Dropdown(sex_options, label="Perpetrator Sex", value=get_random_value(sex_options)),
        gr.Dropdown(race_options, label="Perpetrator Race", value=get_random_value(race_options)),
        gr.Dropdown(ethnicity_options, label="Perpetrator Ethnicity", value=get_random_value(ethnicity_options)),
        gr.Dropdown(agency_type_options, label="Agency Type", value=get_random_value(agency_type_options)),
        gr.Dropdown(relationship_category_options, label="Relationship Category", value=get_random_value(relationship_category_options)),
        gr.Number(label="Victim Age", value=random.randint(10, 80))
    ],
    outputs=gr.Label(num_top_classes=2),
    title="Weapon Category Predictor",
    description="Predict whether the weapon used is a Firearm or Non-Firearm based on various input features."
)

# Launch the interface
##interface.launch()
interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
