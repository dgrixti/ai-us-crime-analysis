import kagglehub
import pandas as pd
import os

# Step 1: Download the latest version using kagglehub
path = kagglehub.dataset_download("mrayushagrawal/us-crime-dataset")
print("Path to dataset files:", path)

# Step 2: Define the path to the downloaded dataset (assuming it's named 'US_Crime_DataSet.csv')
original_file = os.path.join(path, "US_Crime_DataSet.csv")

# Step 3: Load the dataset and split into two parts
df = pd.read_csv(original_file)

# Split the dataset into two files (e.g., half the rows in each)
split_index = len(df) // 2
df1 = df.iloc[:split_index]
df2 = df.iloc[split_index:]

# Save the two smaller files as dataset1.csv and dataset2.csv
df1.to_csv("dataset1.csv", index=False)
df2.to_csv("dataset2.csv", index=False)

print("Dataset split into 'dataset1.csv' and 'dataset2.csv'.")
