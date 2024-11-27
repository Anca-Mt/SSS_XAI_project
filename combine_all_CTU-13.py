import pandas as pd
import os

directory = "datasets_raw/CTU-13"

dataframes = []

for filename in os.listdir(directory):
    if filename.endswith(".parquet"):
        filepath = os.path.join(directory, filename)
        df = pd.read_parquet(filepath)
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_parquet("datasets_raw/CTU-13/CTU-13.parquet")

print("All files have been combined and saved as 'CTU-13.parquet'")