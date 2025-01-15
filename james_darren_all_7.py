import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import os
import random

# Load datasets
df_encoded = pd.read_csv('clean_df_3_single_weapon.csv')
df_dummy = pd.read_csv('crime_dumscalab.csv')

# Initialize encoders for encoded dataset
categorical_columns = ['Agency Type', 'Victim Sex', 'Victim Age', 'Victim Ethnicity',
                      'Perpetrator Sex', 'Perpetrator Ethnicity',
                      'Relationship Category', 'Region', 'Season']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

target_encoder = LabelEncoder()
df_encoded['Weapon Category'] = target_encoder.fit_transform(df_encoded['Weapon Category'])

# Prepare features and target for encoded dataset
X_encoded = df_encoded.drop(columns=['Weapon Category'])
y_encoded = df_encoded['Weapon Category']

# Scale features for encoded dataset
scaler_encoded = StandardScaler()
X_scaled_encoded = scaler_encoded.fit_transform(X_encoded)

# Train-test split for encoded dataset
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(
    X_scaled_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Apply SMOTE for encoded dataset
smote_encoded = SMOTE(random_state=42)
X_train_resampled_encoded, y_train_resampled_encoded = smote_encoded.fit_resample(X_train_encoded, y_train_encoded)

# Prepare data for dummy variables (logistic regression)
scaler_dummy = StandardScaler()
df_dummy['Victim Age Scaled'] = scaler_dummy.fit_transform(df_dummy[['Victim Age']])
X_dummy = df_dummy.drop(columns=['Unnamed: 0', 'Victim Age', 'Weapon Category', 'Weapon Category.1'])
y_dummy = df_dummy['Weapon Category.1']

# Train-test split for dummy dataset
X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
    X_dummy, y_dummy, test_size=0.3, random_state=42, stratify=y_dummy
)

# Apply SMOTE for dummy dataset
smote_dummy = SMOTE(random_state=42)
X_train_resampled_dummy, y_train_resampled_dummy = smote_dummy.fit_resample(X_train_dummy, y_train_dummy)

# File path for saving/loading the model
neural_network_model_path = "neural_network_model.h5"

# Model training functions
def train_or_load_neural_network():
    if os.path.exists(neural_network_model_path):
        print("Loading saved neural network model...")
        model = load_model(neural_network_model_path)
    else:
        print("Training a new neural network model...")
        model = Sequential([
            Dense(16, input_dim=X_train_encoded.shape[1], activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_resampled_encoded, y_train_resampled_encoded, 
                  epochs=50, batch_size=8, validation_split=0.2, verbose=0)
        model.save(neural_network_model_path)
    return model

def train_random_forest():
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, random_state=42, class_weight='balanced'
    )
    rf_model.fit(X_train_resampled_encoded, y_train_resampled_encoded)
    return rf_model

def train_logistic_regression():
    log_model = LogisticRegression(
        C=0.15808394808776696, class_weight=None,
        max_iter=3000, penalty='l1', solver='liblinear'
    )
    log_model.fit(X_train_resampled_dummy, y_train_resampled_dummy)
    return log_model

def train_xgboost():
    scale_pos_weight = len(y_train_resampled_encoded[y_train_resampled_encoded == 0]) / \
                       len(y_train_resampled_encoded[y_train_resampled_encoded == 1])
    xgb_model = XGBClassifier(
        use_label_encoder=False, eval_metric='aucpr',
        random_state=42, scale_pos_weight=scale_pos_weight
    )
    xgb_model.fit(X_train_resampled_encoded, y_train_resampled_encoded)
    return xgb_model

# Helper functions
def format_confusion_matrix(matrix):
    return (
        f"Confusion Matrix:\n"
        f"                    Predicted: Non-Firearm    Predicted: Firearm\n"
        f"Actual: Non-Firearm {matrix[0][0]:<6}                    {matrix[0][1]:<6}\n"
        f"Actual: Firearm     {matrix[1][0]:<6}                    {matrix[1][1]:<6}"
    )

def get_random_value(choices):
    return random.choice(choices)

# Mapping dictionaries for dummy variables
region_mapping = {"Midwest": [0, 0, 0], "Northeast": [1, 0, 0], "South": [0, 1, 0], "West": [0, 0, 1]}
season_mapping = {"Autumn": [0, 0, 0], "Spring": [1, 0, 0], "Summer": [0, 1, 0], "Winter": [0, 0, 1]}
relationship_mapping = {"Acquaintance": [0, 0, 0], "Family": [1, 0, 0], "Lover": [0, 1, 0], "Stranger": [0, 0, 1]}
agency_mapping = {"Municipal Police": [0, 0], "Other Police": [1, 0], "Sheriff": [0, 1]}
sex_mapping = {"Male": 1, "Female": 0}
ethnicity_mapping = {"Hispanic": 0, "Not Hispanic": 1}

# Gradio interface function
def gradio_interface(
    model_choice,
    region, season, relationship, agency,
    victim_sex, perpetrator_sex, victim_ethnicity, perpetrator_ethnicity,
    victim_age
):
    if model_choice == "Logistic Regression":
        # Process input for logistic regression (dummy variables)
        region_dummies = region_mapping[region]
        season_dummies = season_mapping[season]
        relationship_dummies = relationship_mapping[relationship]
        agency_dummies = agency_mapping[agency]
        
        victim_sex_numeric = sex_mapping[victim_sex]
        perpetrator_sex_numeric = sex_mapping[perpetrator_sex]
        victim_ethnicity_numeric = ethnicity_mapping[victim_ethnicity]
        perpetrator_ethnicity_numeric = ethnicity_mapping[perpetrator_ethnicity]
        
        scaled_age = scaler_dummy.transform([[victim_age]])[0][0]
        
        input_features = (
            region_dummies +
            season_dummies +
            relationship_dummies +
            agency_dummies +
            [victim_sex_numeric, perpetrator_sex_numeric,
             victim_ethnicity_numeric, perpetrator_ethnicity_numeric] +
            [scaled_age]
        )
        
        model = train_logistic_regression()
        X_test = X_test_dummy
        y_test = y_test_dummy
        
    else:
        # Process input for other models (encoded variables)
        input_features = [
            label_encoders['Agency Type'].transform([agency])[0],
            label_encoders['Victim Sex'].transform([victim_sex])[0],
            label_encoders['Victim Age'].transform([str(victim_age)])[0],
            label_encoders['Victim Ethnicity'].transform([victim_ethnicity])[0],
            label_encoders['Perpetrator Sex'].transform([perpetrator_sex])[0],
            label_encoders['Perpetrator Ethnicity'].transform([perpetrator_ethnicity])[0],
            label_encoders['Relationship Category'].transform([relationship])[0],
            label_encoders['Region'].transform([region])[0],
            label_encoders['Season'].transform([season])[0]
        ]

        input_features_scaled = scaler_encoded.transform([input_features])

        if model_choice == "Neural Network":
            model = train_or_load_neural_network()
            input_pred_proba = model.predict(input_features_scaled)
            input_pred = (input_pred_proba > 0.5).astype(int).flatten()
            input_pred_proba = np.array([1 - input_pred_proba[0][0], input_pred_proba[0][0]])
        elif model_choice == "Random Forest Classifier":
            model = train_random_forest()
        elif model_choice == "XGBoost":
            model = train_xgboost()
        
        X_test = X_test_encoded
        y_test = y_test_encoded

    # Make predictions
    if model_choice != "Neural Network":
        input_pred = model.predict([input_features])[0]
        input_pred_proba = model.predict_proba([input_features])[0]

    # Get model performance metrics
    if model_choice == "Neural Network":
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred)

    # Format prediction label
    if model_choice == "Logistic Regression":
        prediction_label = "Firearm" if input_pred == 1 else "Non-Firearm"
    else:
        prediction_label = target_encoder.inverse_transform([input_pred])[0]

    return (
        f"Prediction: {prediction_label}",
        f"Probability of Firearm Being Used: {float(input_pred_proba[1]):.2f}",
        f"Probability of Firearm Not Being Used: {float(input_pred_proba[0]):.2f}",
        f"{format_confusion_matrix(conf_matrix)}",
        f"Accuracy: {accuracy:.2f}",
        f"Recall: {recall:.2f}",
        f"Precision: {precision:.2f}",
        f"F1 Score: {f1:.2f}",
        f"Classification Report:\n{class_report}"
    )

# Custom CSS
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

# Function to generate a random example
def generate_random_example():
    return [
        get_random_value(["Neural Network", "Random Forest Classifier", "Logistic Regression", "XGBoost"]),
        get_random_value(["Midwest", "Northeast", "South", "West"]),
        get_random_value(["Autumn", "Spring", "Summer", "Winter"]),
        get_random_value(["Acquaintance", "Family", "Lover", "Stranger"]),
        get_random_value(["Municipal Police", "Other Police", "Sheriff"]),
        get_random_value(["Male", "Female"]),
        get_random_value(["Male", "Female"]),
        get_random_value(["Hispanic", "Not Hispanic"]),
        get_random_value(["Hispanic", "Not Hispanic"]),
        random.randint(0, 100)
    ]

# Create Gradio interface
demo = gr.Interface(
    title="Weapon Use Prediction (Combined Models)",
    description="**Instructions:** Select the model and input the required information.",
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(
            ["Neural Network", "Random Forest Classifier", "Logistic Regression", "XGBoost"],
            value="Neural Network",
            label="Model Choice"
        ),
        gr.Dropdown(["Midwest", "Northeast", "South", "West"], value="Midwest", label="Region"),
        gr.Dropdown(["Autumn", "Spring", "Summer", "Winter"], value="Autumn", label="Season"),
        gr.Dropdown(["Acquaintance", "Family", "Lover", "Stranger"], value="Acquaintance", label="Relationship"),
        gr.Dropdown(["Municipal Police", "Other Police", "Sheriff"], value="Municipal Police", label="Agency"),
        gr.Dropdown(["Male", "Female"], value="Male", label="Victim Sex"),
        gr.Dropdown(["Male", "Female"], value="Male", label="Perpetrator Sex"),
        gr.Dropdown(["Hispanic", "Not Hispanic"], value="Hispanic", label="Victim Ethnicity"),
        gr.Dropdown(["Hispanic", "Not Hispanic"], value="Hispanic", label="Perpetrator Ethnicity"),
        gr.Slider(0, 100, step=1, value=30, label="Victim Age")
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
        gr.Textbox(label="Classification Report")
    ],
    css=custom_css,
    examples=[generate_random_example() for _ in range(5)]  # Generate 5 random examples as lists
)

# Launch the interface
demo.launch()
