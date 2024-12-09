import gradio as gr
import pandas as pd
import joblib
import random

# Load the saved model and preprocessor
saved_objects = joblib.load('relationship_model.pkl')
model = saved_objects['model']
preprocessor = saved_objects['preprocessor']

# Dropdown options
state_options = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California']
month_options = ['January', 'February', 'March', 'April', 'May']
victim_sex_options = ['Male', 'Female']
victim_race_options = ['White', 'Black', 'Asian/Pacific Islander']
victim_ethnicity_options = ['Hispanic', 'Not Hispanic']
perp_sex_options = ['Male', 'Female']
perp_race_options = ['White', 'Black', 'Asian/Pacific Islander']
perp_ethnicity_options = ['Hispanic', 'Not Hispanic']
agency_type_options = ['Municipal Police', 'Sheriff']
weapon_category_options = ['Firearm', 'Non-Firearm']

# Define prediction function
def predict_relationship(state, month, victim_sex, victim_race, victim_ethnicity,
                          perpetrator_sex, perpetrator_race, perpetrator_ethnicity,
                          agency_type, weapon_category, victim_age):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([{
        'State': state,
        'Month': month,
        'Victim Sex': victim_sex,
        'Victim Race': victim_race,
        'Victim Ethnicity': victim_ethnicity,
        'Perpetrator Sex': perpetrator_sex,
        'Perpetrator Race': perpetrator_race,
        'Perpetrator Ethnicity': perpetrator_ethnicity,
        'Agency_Type_grouped': agency_type,
        'Weapon Category': weapon_category,
        'Victim Age': victim_age
    }])

    # Transform the input data using the pre-fitted preprocessor
    input_transformed = preprocessor.transform(input_data)

    # Predict probabilities
    probabilities = model.predict_proba(input_transformed)[0]

    # Map probabilities to class labels as floats
    class_labels = model.classes_
    probabilities_dict = {label: prob for label, prob in zip(class_labels, probabilities)}

    return probabilities_dict

# Create Gradio interface
interface = gr.Interface(
    fn=predict_relationship,
    inputs=[
        gr.Dropdown(state_options, label="State", value=random.choice(state_options)),
        gr.Dropdown(month_options, label="Month", value=random.choice(month_options)),
        gr.Dropdown(victim_sex_options, label="Victim Sex", value=random.choice(victim_sex_options)),
        gr.Dropdown(victim_race_options, label="Victim Race", value=random.choice(victim_race_options)),
        gr.Dropdown(victim_ethnicity_options, label="Victim Ethnicity", value=random.choice(victim_ethnicity_options)),
        gr.Dropdown(perp_sex_options, label="Perpetrator Sex", value=random.choice(perp_sex_options)),
        gr.Dropdown(perp_race_options, label="Perpetrator Race", value=random.choice(perp_race_options)),
        gr.Dropdown(perp_ethnicity_options, label="Perpetrator Ethnicity", value=random.choice(perp_ethnicity_options)),
        gr.Dropdown(agency_type_options, label="Agency Type", value=random.choice(agency_type_options)),
        gr.Dropdown(weapon_category_options, label="Weapon Category", value=random.choice(weapon_category_options)),
        gr.Number(label="Victim Age", value=random.randint(18, 80)),  # Random age between 18 and 80
    ],
    outputs=gr.Label(num_top_classes=4),  # Display top 4 probabilities
    title="Relationship Category Predictor",
    description="Predict the likely relationship between victim and perpetrator based on input features."
)

# Launch the interface
interface.launch()
