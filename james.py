# Import necessary libraries
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Import SMOTE for handling imbalanced data


# Load your dataset (assuming 'crime.csv' file is available)
crime = pd.read_csv('07. crime_dataset_analysis_v2.csv')
crime = pd.DataFrame(crime)
## age is still between 0 and 100

# Create a scaler instance
scaler = StandardScaler()

# Scale the 'Victim Age' for model training
crime['Victim Age Scaled'] = scaler.fit_transform(crime[['Victim Age']])

# Prepare the model's feature data (X) and target variable (y)
X_dummy = crime.drop(columns=['Victim Age', 'Weapon Category'])
y_dummy = crime['Weapon Category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.3, random_state=42, stratify=y_dummy)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)  # Initialize SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # Oversample the minority class

# Update Logistic Regression model with best parameters
best_params = {
    'C': 0.15808394808776696,
    'class_weight': None,
    'max_iter': 3000,
    'penalty': 'l1',
    'solver': 'liblinear'
}

# Initialize and train the Logistic Regression model
model = LogisticRegression(**best_params)
model.fit(X_train_resampled, y_train_resampled)

# Helper function to convert "Yes"/"No" to numeric
def convert_input(value):
    return 1 if value == "Yes" else 0

# Scale victim age (same scaling as done in the dataset) for the interface
def scale_victim_age(victim_age):
    return scaler.transform([[victim_age]])[0][0]  # Scale based on the previously fitted scaler

# Function to format confusion matrix (improved clarity)
def format_confusion_matrix(matrix):
    return (
        f"Confusion Matrix:\n"
        f"               Predicted: No     Predicted: Yes\n"
        f"Actual: No    {matrix[0][0]:<6}  {matrix[0][1]:<6}\n"
        f"Actual: Yes   {matrix[1][0]:<6}  {matrix[1][1]:<6}"
    )

# Evaluate the model on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')  # Handle multi-class if applicable
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(format_confusion_matrix(cm))


# Gradio interface function with simplified dropdowns
def gradio_interface(
    region, season, relationship, agency,
    victim_sex, perpetrator_sex, ethnicity, victim_age
):
    # Define mapping for dropdowns to dummy variables
    region_mapping = {"Midwest": [0, 0, 0], "Northeast": [1, 0, 0], "South": [0, 1, 0], "West": [0, 0, 1]}
    season_mapping = {"Autumn": [0, 0, 0], "Spring": [1, 0, 0], "Summer": [0, 1, 0], "Winter": [0, 0, 1]}
    relationship_mapping = {"Acquaintance": [0, 0, 0], "Family": [1, 0, 0], "Lover": [0, 1, 0], "Stranger": [0, 0, 1]}
    agency_mapping = {"Municipal Police": [0, 0], "Other Police": [1, 0], "Sheriff": [0, 1]}
    sex_mapping = {"Male": 1, "Female": 0}
    ethnicity_mapping = {"Hispanic": 0, "Not Hispanic": 1}

    # Map selected values to dummy variables
    region_dummies = region_mapping[region]
    season_dummies = season_mapping[season]
    relationship_dummies = relationship_mapping[relationship]
    agency_dummies = agency_mapping[agency]

    # Map sex and ethnicity to numeric values
    victim_sex_numeric = sex_mapping[victim_sex]
    perpetrator_sex_numeric = sex_mapping[perpetrator_sex]
    ethnicity_numeric = ethnicity_mapping[ethnicity]

    # Compile input features
    input_features = (
        region_dummies +
        season_dummies +
        relationship_dummies +
        agency_dummies +
        [victim_sex_numeric, perpetrator_sex_numeric, ethnicity_numeric] +
        [scale_victim_age(victim_age)]  # Scale victim age for the model
    )

    # Model predictions
    input_pred = model.predict([input_features])[0]  # Single prediction
    input_pred_proba = model.predict_proba([input_features])[0]  # Probabilities for both classes

    # Map numeric prediction to labels
    prediction_label = "Firearm" if input_pred == 1 else "Non-Firearm"

    # Compute evaluation metrics
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Improved confusion matrix format
    formatted_conf_matrix = format_confusion_matrix(conf_matrix)

    return (
        f"Prediction: {prediction_label}",
        f"Probability of Firearm Being Used: {input_pred_proba[1]:.2f}",
        f"Probability of Firearm Not Being Used: {input_pred_proba[0]:.2f}",
        f"{formatted_conf_matrix}",
        f"Accuracy: {accuracy:.2f}",
        f"Recall: {recall:.2f}",
        f"Precision: {precision:.2f}",
        f"F1 Score: {f1:.2f}",
        f"Classification Report:\n{class_report}"
    )

# Custom CSS for the Gradio interface
custom_css = """
body {
    background: linear-gradient(to bottom right, #6a11cb, #2575fc);
    color: white;
    font-family: Arial, sans-serif;
}
.error-text {
    color: red;
    font-weight: bold;
}
"""

# Gradio interface with simplified dropdowns
gr.Interface(
    title="Weapon Use Prediction",
    description="**Instructions:** Select one option per category.",
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Midwest", "Northeast", "South", "West"], value="Midwest", label="Region"),
        gr.Dropdown(["Autumn", "Spring", "Summer", "Winter"], value="Autumn", label="Season"),
        gr.Dropdown(["Acquaintance", "Family", "Lover", "Stranger"], value="Acquaintance", label="Relationship"),
        gr.Dropdown(["Municipal Police", "Other Police", "Sheriff"], value="Municipal Police", label="Agency"),
        gr.Dropdown(["Male", "Female"], value="Male", label="Victim Sex"),
        gr.Dropdown(["Male", "Female"], value="Male", label="Perpetrator Sex"),
        gr.Dropdown(["Hispanic", "Not Hispanic"], value="Hispanic", label="Ethnicity"),
        gr.Slider(0, 100, step=1, value=0, label="Victim Age"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability of Firearm Being Used"),
        gr.Textbox(label="Probability of Firearm Not Being Used"),
        gr.Textbox(label="Confusion Matrix"),
        gr.Textbox(label="Accuracy"),
        gr.Textbox(label="Recall"),
        gr.Textbox(label="Precision"),
        gr.Textbox(label="F1 Score"),
        gr.Textbox(label="Classification Report"),
    ],
    css=custom_css,
).launch()