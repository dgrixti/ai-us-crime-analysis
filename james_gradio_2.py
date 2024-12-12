import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr

cr = pd.read_csv('ML_REVISED_V6_noyear_nounknowns.csv')

# Group states into North, South, West & East
state_to_region = {
    'Rhodes Island': 'Northeast',
    'District of Columbia': 'Northeast',
    'Maine': 'Northeast', 
    'New Hampshire': 'Northeast', 
    'Vermont': 'Northeast', 
    'Massachusetts': 'Northeast', 
    'Rhode Island': 'Northeast', 
    'Connecticut': 'Northeast',
    'New York': 'Northeast', 
    'New Jersey': 'Northeast', 
    'Pennsylvania': 'Northeast',
    'Delaware': 'South', 
    'Maryland': 'South', 
    'Virginia': 'South', 
    'North Carolina': 'South',
    'South Carolina': 'South', 
    'Georgia': 'South', 
    'Florida': 'South', 
    'Alabama': 'South',
    'Mississippi': 'South', 
    'Louisiana': 'South', 
    'Arkansas': 'South', 
    'Tennessee': 'South',
    'Kentucky': 'South', 
    'West Virginia': 'South', 
    'Oklahoma': 'South', 
    'Texas': 'South',
    'Ohio': 'Midwest', 
    'Indiana': 'Midwest', 
    'Illinois': 'Midwest', 
    'Iowa': 'Midwest', 
    'Missouri': 'Midwest', 
    'Michigan': 'Midwest', 
    'Wisconsin': 'Midwest', 
    'Minnesota': 'Midwest',
    'North Dakota': 'Midwest', 
    'South Dakota': 'Midwest', 
    'Nebraska': 'Midwest', 
    'Kansas': 'Midwest',
    'Montana': 'West', 
    'Wyoming': 'West', 
    'Idaho': 'West', 
    'Washington': 'West', 
    'Oregon': 'West',
    'California': 'West', 
    'Nevada': 'West', 
    'Utah': 'West', 
    'Colorado': 'West', 
    'Arizona': 'West',
    'New Mexico': 'West', 
    'Alaska': 'West', 
    'Hawaii': 'West'
}

cr['Region'] = cr['State'].replace(state_to_region)

# Drop the 'State' column if no longer needed
cr.drop('State', axis=1, inplace=True)

# Define a dictionary mapping months to seasons
month_to_season = {
    'January': 'Winter', 'February': 'Winter', 'December': 'Winter',
    'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
    'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
    'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn'
}

# Replace the 'Month' column with the corresponding 'Season'
cr['Season'] = cr['Month'].map(month_to_season)

# Optionally, drop the 'Month' column if no longer needed
cr.drop('Month', axis=1, inplace=True)

# Check the result
cr = pd.DataFrame(cr)
cr.info()

# List features to convert to categorical
categ_features = [
    'Region', 'Season', 'Victim Sex', 'Victim Race', 'Victim Ethnicity',
    'Perpetrator Sex', 'Perpetrator Race', 'Perpetrator Ethnicity',
    'Relationship Category', 'Agency_Type_grouped'
]

# Convert features to categorical (binary)
cr[categ_features] = cr[categ_features].astype('category')

# Target variable to convert to categorical
cr['Weapon Category'] = cr['Weapon Category'].astype('category')

# Drop redundant columns
cr.drop(columns=['Victim Race', 'Perpetrator Race'], inplace=True)

# Update categorical features list
categ_features = [
    'Region', 'Season', 'Victim Sex', 'Victim Ethnicity',
    'Perpetrator Sex', 'Perpetrator Ethnicity',
    'Relationship Category', 'Agency_Type_grouped'
]

# Apply get_dummies to all categorical features
non_sex_ethnicity_features = [
    col for col in categ_features 
    if col not in ['Victim Sex', 'Perpetrator Sex', 'Victim Ethnicity', 'Perpetrator Ethnicity']
]

# Apply get_dummies
cr_non_sex_ethnicity = pd.get_dummies(cr[non_sex_ethnicity_features], drop_first=True)
victim_sex_dummies = pd.get_dummies(cr['Victim Sex'], prefix='Victim Sex', drop_first=True)
perpetrator_sex_dummies = pd.get_dummies(cr['Perpetrator Sex'], prefix='Perpetrator Sex', drop_first=True)
victim_ethnicity_dummies = pd.get_dummies(cr['Victim Ethnicity'], prefix='Victim Ethnicity', drop_first=True)
perpetrator_ethnicity_dummies = pd.get_dummies(cr['Perpetrator Ethnicity'], prefix='Perpetrator Ethnicity', drop_first=True)

# Concatenate all the dummy variable DataFrames
cr = pd.concat([
    cr, cr_non_sex_ethnicity, victim_sex_dummies, perpetrator_sex_dummies,
    victim_ethnicity_dummies, perpetrator_ethnicity_dummies
], axis=1)

# Drop original categorical columns
cr.drop(
    columns=[
        'Victim Sex', 'Perpetrator Sex', 'Victim Ethnicity', 'Perpetrator Ethnicity',
        'Season', 'Region', 'Relationship Category', 'Agency_Type_grouped'
    ], inplace=True
)

# Print the updated DataFrame
print(cr.head())

# Prepare the final DataFrame for analysis
analysis_data = cr.copy()

# Check each feature for Complete Separation
categorical_columns = [
    col for col in analysis_data.columns 
    if col != 'Victim Age' and col != 'Weapon Category'
]

for feature in categorical_columns:
    crosstab = pd.crosstab(analysis_data[feature], analysis_data['Weapon Category'])
    print(f"Cross-tabulation for {feature}:\n{crosstab}\n")

# Map 'Firearm' to 1 and 'Non-Firearm' to 0
analysis_data['Weapon Category'] = analysis_data['Weapon Category'].map({'Firearm': 1, 'Non-Firearm': 0})

# Combine ethnicity columns into a binary feature
analysis_data['Ethnicity_Not_Hispanic_Combined'] = (
    (analysis_data['Victim Ethnicity_Not Hispanic'] == 1) |
    (analysis_data['Perpetrator Ethnicity_Not Hispanic'] == 1)
).astype(int)

# Drop original ethnicity columns
analysis_data.drop(columns=['Victim Ethnicity_Not Hispanic', 'Perpetrator Ethnicity_Not Hispanic'], inplace=True)

# Check distribution of the combined variable
print(analysis_data['Ethnicity_Not_Hispanic_Combined'].value_counts())

# Start by scaling Victim Age (with mean 0 and variance 1)
from sklearn.preprocessing import StandardScaler

# Create a scaler instance
scaler = StandardScaler()

# Scale Victim Age
analysis_data['Victim Age Scaled'] = scaler.fit_transform(analysis_data[['Victim Age']])

# Drop the original 'age' column
analysis_data.drop(columns=['Victim Age'], inplace=True)

# Check the updated DataFrame
print(analysis_data.head())

# Recalculate VIF after scaling Victim Age
from statsmodels.stats.outliers_influence import variance_inflation_factor

features_for_vif = analysis_data.drop(columns=['Weapon Category'])  # Exclude target variable
features_for_vif = features_for_vif.astype(int)
features_for_vif['Victim Age Scaled'] = analysis_data['Victim Age Scaled']

# Define X and y
X = features_for_vif
y = analysis_data['Weapon Category']

# Regularization techniques - Lasso & Ridge Regression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_model(penalty, cv, random_state):
    model = LogisticRegressionCV(
        penalty=penalty,
        solver='saga' if penalty == 'l1' else 'lbfgs',
        cv=cv,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    return f"Model - Penalty: {penalty}, CV: {cv}, Random State: {random_state}\nAccuracy: {accuracy:.4f}, AUC: {auc:.4f}"

import gradio as gr

def interface(penalty, cv, random_state):
    try:
        random_state = int(random_state)
        cv = int(cv)
        # Convert user-friendly dropdown values to actual penalty values
        penalty_mapping = {"Lasso": "l1", "Ridge": "l2"}
        penalty_value = penalty_mapping[penalty]
        return train_model(penalty_value, cv, random_state)
    except Exception as e:
        return f"Error: {str(e)}"

penalty_options = ["Lasso", "Ridge"]
cv_options = [4, 5]

interface_gr = gr.Interface(
    fn=interface,
    inputs=[
        gr.Dropdown(choices=penalty_options, label="Penalty (Lasso or Ridge)"),
        gr.Dropdown(choices=cv_options, label="Cross-Validation (CV)"),
        gr.Textbox(label="Random State", value="42")
    ],
    outputs="text",
    title="Lasso vs Ridge Logistic Regression",
    description="Choose between Lasso or Ridge regularization, CV folds, and random state to train a model."
)

interface_gr.launch()
