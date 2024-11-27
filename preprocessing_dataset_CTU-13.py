import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def map_labels(label):
    if "Normal" in label:
        return 0  # Benign
    elif "Botnet" in label:
        return 1  # Malicious
    else:
        return None  # Background or irrelevant traffic

def apply_binary_label(data):
    data = data.copy()
    data['BinaryLabel'] = data.loc[:, 'label'].apply(map_labels)
    # Drop rows where the label is 'None' (Background or irrelevant)
    data = data.dropna(subset=['BinaryLabel'])

    data['BinaryLabel'] = data['BinaryLabel'].astype(int)

    print("The distribution of labels is: \n", data['BinaryLabel'].value_counts())

    return data

def encode_features(data, features):
    encoder = LabelEncoder()
    for f in features:
        data[f] = encoder.fit_transform(data[f])
    return data

def clean_data(data):
    """
    Drops the columns 'Family', 'dir', and 'label' from the dataset.
    Delete rows that have one value missing
    """
    columns_to_drop = ['Family', 'dir', 'label']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    data = data.dropna()

    return data

if __name__ == '__main__':

    CTU_13 = pd.read_parquet("datasets_raw/CTU-13/CTU-13.parquet")
    CTU_13 = apply_binary_label(CTU_13)
    CTU_13 = encode_features(CTU_13, ['proto', 'state'])
    CTU_13 = clean_data(CTU_13)
    
    os.makedirs("datasets_csv/CTU-13", exist_ok=True)
    CTU_13.to_csv("datasets_csv/CTU-13/CTU-13.csv", index=False)

    print(CTU_13.head())
