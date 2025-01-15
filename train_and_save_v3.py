import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import joblib

# Load the dataset
data = pd.read_csv("resampled_training_data_human_readable.csv")

# Features (including 'Relationship Category') and Target ('Weapon Category')
X = data.drop(['Weapon Category'], axis=1)  # Features
y = data['Weapon Category']  # Target

# Identify categorical columns
cat_features = X.select_dtypes(include=['object']).columns

# Preprocessor: OneHotEncode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'  # Keep numerical features unchanged
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Transform features
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_transformed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_transformed)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and preprocessor
joblib.dump({'model': model, 'preprocessor': preprocessor}, 'weapon_category_model.pkl')
print("Model and preprocessor saved to 'weapon_category_model.pkl'")
print("VVV3")