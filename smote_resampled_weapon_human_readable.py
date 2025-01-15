import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv("master_data_clean.csv")

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

# Transform the features
X_transformed = preprocessor.fit_transform(X)

# Check the shape of X after preprocessing
print("Shape of features after preprocessing (X):", X_transformed.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution before and after SMOTE
print("Original training set class distribution:", Counter(y_train))
print("Resampled training set class distribution:", Counter(y_train_resampled))
print("Shape of resampled training features (X_train_resampled):", X_train_resampled.shape)

# Convert the resampled data to a dense format if necessary
if hasattr(X_train_resampled, "toarray"):
    X_train_resampled = X_train_resampled.toarray()

# Decode one-hot encoded categorical features back to original categories
encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
decoded_categorical = pd.DataFrame(
    preprocessor.named_transformers_['cat'].inverse_transform(X_train_resampled[:, :len(encoded_cat_features)]),
    columns=cat_features
)

# Extract numerical features
numerical_start_idx = len(encoded_cat_features)
numerical_data = pd.DataFrame(
    X_train_resampled[:, numerical_start_idx:],
    columns=num_features
)

# Combine decoded categorical features and numerical features
X_resampled_human_readable = pd.concat([decoded_categorical, numerical_data], axis=1)

# Add the target column
X_resampled_human_readable['Weapon Category'] = y_train_resampled.reset_index(drop=True)

# Save the human-readable resampled dataset to a new CSV file
X_resampled_human_readable.to_csv("resampled_training_data_human_readable_v3.csv", index=False)

print("Resampled data saved to 'resampled_training_data_human_readable_v3.csv'")
