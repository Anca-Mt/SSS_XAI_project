from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def arff_to_df(arff_path):
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)

    # From binary to strings in df
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    return df


def map_labels(label):
    if "normal" in label:
        return 0
    elif "anomaly" in label:
        return 1


def apply_binary_label(data):
    data = data.copy()
    data['BinaryClass'] = data.loc[:, 'class'].apply(map_labels)
    # Drop rows where the label is 'None' (Background or irrelevant)
    data = data.dropna(subset=['BinaryClass'])
    data = data.drop(columns="class", errors='ignore')
    data['BinaryClass'] = data['BinaryClass'].astype(int)

    print("The distribution of labels is: \n", data['BinaryClass'].value_counts())

    return data

def encode_features(data, features):
    encoder = LabelEncoder()
    for f in features:
        data[f] = encoder.fit_transform(data[f])
    return data


if __name__ == '__main__':
    KDD_train_raw_path = "datasets_raw/NSL-KDD/KDDTrain+.arff"
    KDD_test_raw_path = "datasets_raw/NSL-KDD/KDDTest+.arff"

    df_train = arff_to_df(KDD_train_raw_path)
    df_test = arff_to_df(KDD_test_raw_path)

    df_train = apply_binary_label(df_train)
    df_test = apply_binary_label(df_test)

    features_to_encode = ['protocol_type', 'service', 'flag']
    df_train = encode_features(df_train, features_to_encode)
    df_test = encode_features(df_test, features_to_encode)

    df_train.to_csv("datasets_csv/NSL-KDD/NSL-KDDTrain.csv", index=False)
    df_test.to_csv("datasets_csv/NSL-KDD/NSL-KDDTest.csv", index=False)

    print(f"CSV files fro NSL-KDD successfully saved!")