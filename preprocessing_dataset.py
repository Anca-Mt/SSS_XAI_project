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
    # CTU_1 = pd.read_parquet("CTU-13/1-Neris-20110810.binetflow.parquet")
    # CTU_1 = apply_binary_label(CTU_1)
    # CTU_1 = encode_features(CTU_1, ['proto', 'state'])
    # CTU_1 = clean_data(CTU_1)
    #
    # CTU_1.to_csv("CTU-13_csvs/1-Neris-20110810.csv", index=False)

    CTU_13 = pd.read_parquet("CTU-13/combined_CTU-13.parquet")
    CTU_13 = apply_binary_label(CTU_13)
    CTU_13 = encode_features(CTU_13, ['proto', 'state'])
    CTU_13 = clean_data(CTU_13)

    CTU_13.to_csv("CTU-13_csvs/combined_CTU-13.csv", index=False)

    print(CTU_13.head())
