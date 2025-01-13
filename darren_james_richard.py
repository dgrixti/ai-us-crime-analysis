# Import necessary libraries
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load your dataset
crime = pd.read_csv('07. crime_dataset_analysis_v2.csv')
crime = pd.DataFrame(crime)

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
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Function to train and return the Random Forest model
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to train and return the Logistic Regression model
def train_logistic_regression(X_train, y_train):
    best_params = {
        'C': 0.15808394808776696,
        'class_weight': None,
        'max_iter': 3000,
        'penalty': 'l1',
        'solver': 'liblinear'
    }
    log_model = LogisticRegression(**best_params)
    log_model.fit(X_train, y_train)
    return log_model

# Function to train and return the XGBoost model
def train_xgboost(X_train, y_train):
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Scale victim age (same scaling as done in the dataset)
def scale_victim_age(victim_age):
    return scaler.transform([[victim_age]])[0][0]

# Helper function to format confusion matrix
def format_confusion_matrix(matrix):
    return (
        f"Confusion Matrix:\n"
        f"               Predicted: No     Predicted: Yes\n"
        f"Actual: No    {matrix[0][0]:<6}  {matrix[0][1]:<6}\n"
        f"Actual: Yes   {matrix[1][0]:<6}  {matrix[1][1]:<6}"
    )

# Gradio interface function
def gradio_interface(
    model_choice, region, season, relationship, agency,
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
        [scale_victim_age(victim_age)]
    )

    # Train and select the appropriate model
    if model_choice == "Random Forest Classifier":
        model = train_random_forest(X_train_resampled, y_train_resampled)
    elif model_choice == "Logistic Regression":
        model = train_logistic_regression(X_train_resampled, y_train_resampled)
    elif model_choice == "XGBoost":
        model = train_xgboost(X_train_resampled, y_train_resampled)
    else:
        return "Invalid Model Choice", {}, "", "", "", "", "", "", "", ""

    # Model predictions
    input_pred = model.predict([input_features])[0]
    input_pred_proba = model.predict_proba([input_features])[0]

    # Map numeric prediction to labels
    prediction_label = "Firearm" if input_pred == 1 else "Non-Firearm"

    # Compute evaluation metrics
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred)

    return (
        f"Prediction: {prediction_label}",
        {"Firearm": input_pred_proba[1], "Non-Firearm": input_pred_proba[0]},
        f"Probability of Firearm Being Used: {input_pred_proba[1]:.2f}",
        f"Probability of Firearm Not Being Used: {input_pred_proba[0]:.2f}",
        f"{format_confusion_matrix(conf_matrix)}",
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

# Gradio interface
gr.Interface(
    title="Weapon Use Prediction (Select Model)",
    description="**Instructions:** Select the model and one option per category.",
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Random Forest Classifier", "Logistic Regression", "XGBoost"], value="Random Forest Classifier", label="Model Choice"),
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
        gr.Label(num_top_classes=2, label="Prediction Probabilities (Graphical)"),
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
