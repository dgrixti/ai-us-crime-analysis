import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the resampled dataset
data = pd.read_csv("resampled_weapon_training_data.csv")

# Separate features and target
X = data.drop(['Weapon Category'], axis=1)  # Features
y = data['Weapon Category']  # Target

# Split the resampled dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'random_forest_weapon_model.pkl')
print("Model saved to 'random_forest_weapon_model.pkl'")
