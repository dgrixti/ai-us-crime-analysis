import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

# Load the human-readable resampled dataset
data = pd.read_csv("resampled_training_data_human_readable.csv")

# Separate features and target
X = data.drop(['Relationship Category'], axis=1)  # Features
y = data['Relationship Category']  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify categorical and numerical columns
cat_features = X.select_dtypes(include=['object']).columns
num_features = X.select_dtypes(exclude=['object']).columns

# Preprocess categorical variables using OneHotEncoder
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
], remainder='passthrough')  # Keep numerical features unchanged

# Transform the features
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
model.fit(X_train_transformed, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_transformed)
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model to a file
import joblib

#### IMPOORTANT: We need to save the preprocessor data with the model for Gradio to keep consistency

# Save the model and preprocessor together
joblib.dump({'model': model, 'preprocessor': preprocessor}, 'relationship_model.pkl')
print("Model and preprocessor saved to 'relationship_model.pkl'")

