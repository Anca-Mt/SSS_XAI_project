import pandas as pd
import os

# Path to the directory containing the parquet files
directory = "CTU-13"

# List to hold all DataFrames
dataframes = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".parquet"):
        filepath = os.path.join(directory, filename)
        # Read each parquet file
        df = pd.read_parquet(filepath)
        dataframes.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new parquet file
combined_df.to_parquet("CTU-13/combined_CTU-13.parquet")

print("All files have been combined and saved as 'combined_CTU-13.parquet'")