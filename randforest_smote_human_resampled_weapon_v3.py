import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import joblib

# Load the resampled dataset
data = pd.read_csv("resampled_training_data_human_readable_v3.csv")

# Separate features and target
X = data.drop(['Weapon Category'], axis=1)  # Features
y = data['Weapon Category']  # Target

# Identify categorical and numerical columns
cat_features = X.select_dtypes(include=['object']).columns
num_features = X.select_dtypes(exclude=['object']).columns

# Preprocessor: OneHotEncode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'  # Keep numerical features unchanged
)

# Transform features
X_transformed = preprocessor.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and preprocessor
joblib.dump({'model': model, 'preprocessor': preprocessor}, 'weapon_category_model_v3.pkl')
print("Model and preprocessor saved to 'weapon_category_model_v3.pkl'")
