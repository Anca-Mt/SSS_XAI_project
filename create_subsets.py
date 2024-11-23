from cProfile import label

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import os
import argparse

valid_datasets = [
    "CTU-13",
    "NSL-KDD",
    "UNSW-NB15",
]

def split_data_in_3(data, label_column, test_size=0.2, explain_size=0.05):

    X = data.drop(columns=[label_column])
    y = data[label_column]

    # Stratify is used to maintain the same class distribution as in the original data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_test, X_explain, y_test, y_explain = train_test_split(
        X_temp, y_temp, test_size=explain_size, random_state=42, stratify=y_temp
    )
    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "explain": (X_explain, y_explain)
    }

def split_data_in_2(data, label_column, explain_size=0.05):
    X = data.drop(columns=[label_column])
    y = data[label_column]

    # Stratify is used to maintain the same class distribution as in the original data
    X_test, X_explain, y_test, y_explain = train_test_split(
        X, y, test_size=explain_size, random_state=42, stratify=y
    )

    return {
        "test": (X_test, y_test),
        "explain": (X_explain, y_explain)
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Splits given dataset to train, test and explain datasets.')

    argparser.add_argument('-d', '--dataset', default='NSL-KDD', type=str, help='The datset to split as a string')

    args = argparser.parse_args()

    if args.dataset is None:
        raise ValueError(f"Dataset argument not provided. Use the -d or --dataset flag to provide one of {valid_datasets}.")

    if args.dataset == 'NSL-KDD':
        dataset_csv_train = pd.read_csv(f"datasets_csv/{args.dataset}/{args.dataset}Train.csv")
        dataset_csv_test = pd.read_csv(f"datasets_csv/{args.dataset}/{args.dataset}Test.csv")

        X_train =  dataset_csv_train.drop(columns=['BinaryClass']).values
        y_train = dataset_csv_train['BinaryClass'].values

        splits2 = split_data_in_2(dataset_csv_test, label_column="BinaryClass")
        X_test, y_test = splits2["test"]
        X_explain, y_explain = splits2["explain"]

    else:
        dataset_csv = pd.read_csv(f"datasets_csv/{args.dataset}/{args.dataset}.csv")
        splits3 = split_data_in_3(dataset_csv, label_column="BinaryLabel")

        X_train, y_train = splits3["train"]
        X_test, y_test = splits3["test"]
        X_explain, y_explain = splits3["explain"]

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Explain set: {len(X_explain)} samples")

    os.makedirs(f"datasets_npy/{args.dataset}", exist_ok=True)
    
    np.save(f"datasets_npy/{args.dataset}/X_train.npy", X_train)
    np.save(f"datasets_npy/{args.dataset}/X_test.npy", X_test)
    np.save(f"datasets_npy/{args.dataset}/Y_train.npy", y_train)
    np.save(f"datasets_npy/{args.dataset}/Y_test.npy", y_test)
    np.save(f"datasets_npy/{args.dataset}/X_explain.npy", X_explain)
    np.save(f"datasets_npy/{args.dataset}/Y_explain.npy", y_explain)