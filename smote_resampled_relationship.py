import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv("master_data_clean.csv")

# Separate features and target
X = data.drop(['Relationship Category'], axis=1)  # Drop the target column
y = data['Relationship Category']  # Target column

# Preprocess categorical variables using OneHotEncoder
cat_features = X.select_dtypes(include=['object']).columns
num_features = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
], remainder='passthrough')  # Keep numerical features unchanged

# Transform the features
X = preprocessor.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution before and after SMOTE
print("Original training set class distribution:", Counter(y_train))
print("Resampled training set class distribution:", Counter(y_train_resampled))

# Convert resampled features to dense if they are sparse
if hasattr(X_train_resampled, "toarray"):
    X_train_resampled = X_train_resampled.toarray()

# Resolve feature names
encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
feature_names = list(encoded_cat_features) + list(num_features)

# Convert the resampled features and target to DataFrame
X_resampled_df = pd.DataFrame(X_train_resampled, columns=feature_names)
y_resampled_df = pd.DataFrame(y_train_resampled, columns=['Relationship Category'])

# Combine features and target into a single DataFrame
resampled_data = pd.concat([X_resampled_df, y_resampled_df], axis=1)

# Save the resampled dataset to a CSV file
resampled_data.to_csv("resampled_relationship_training_data.csv", index=False)
print("Resampled data saved to 'resampled_relationship_training_data.csv'")
