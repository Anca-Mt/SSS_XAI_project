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
    CTU_1 = pd.read_parquet("CTU-13/1-Neris-20110810.binetflow.parquet")
    CTU_1 = apply_binary_label(CTU_1)
    CTU_1 = encode_features(CTU_1, ['proto', 'state'])
    CTU_1 = clean_data(CTU_1)

    CTU_1.to_csv("CTU-13_csvs/1-Neris-20110810.csv", index=False)

    # CTU_3 = pd.read_parquet("C:/Users/User/PYTHON PROJECTS/SSS_A2/xai-pipeline/CTU-13/3-Rbot-20110812.binetflow.parquet")
    # CTU_3.to_csv("C:/Users/User/PYTHON PROJECTS/SSS_A2/xai-pipeline/CTU-13_csvs/3-Rbot-20110812.csv", index=False)
    #
    # CTU_5 = pd.read_parquet("C:/Users/User/PYTHON PROJECTS/SSS_A2/xai-pipeline/CTU-13/5-Virut-20110815-2.binetflow.parquet")
    # CTU_5.to_csv("C:/Users/User/PYTHON PROJECTS/SSS_A2/xai-pipeline/CTU-13_csvs/5-Virut-20110815.csv", index=False)

    print(CTU_1.head())
