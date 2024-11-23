import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_features(train_data, test_data, features):

    for feature in features:
        encoder = LabelEncoder()
        # Combine unique values from both train and test
        all_categories = pd.concat([train_data[feature], test_data[feature]]).unique()
        encoder.fit(all_categories)

        # Transform both datasets
        train_data[feature] = encoder.transform(train_data[feature])
        test_data[feature] = encoder.transform(test_data[feature])
    return train_data, test_data

def clean_data(data):
    columns_to_drop = ['id', 'attack_cat']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    data = data.dropna()

    return data

if __name__ == '__main__':
    KDD_train_raw_path = "datasets_raw/UNSW-NB15/UNSW_NB15_training-set.csv"
    KDD_test_raw_path = "datasets_raw/UNSW-NB15/UNSW_NB15_testing-set.csv"

    df_train = pd.read_csv(KDD_train_raw_path)
    df_test = pd.read_csv(KDD_test_raw_path)

    features_to_encode = ['proto', 'service', 'state', 'attack_cat']
    df_train, df_test = encode_features(df_train, df_test, features_to_encode)

    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    df_train.to_csv("datasets_csv/UNSW-NB15/UNSW-NB15Train.csv", index=False)
    df_test.to_csv("datasets_csv/UNSW-NB15/UNSW-NB15Test.csv", index=False)

    print(f"CSV files for UNSW-NB15 successfully saved!")