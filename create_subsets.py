from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def split_data(data, label_column, test_size=0.2, explain_size=0.05):

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


if __name__ == "__main__":

    CTU_13 = pd.read_csv("CTU-13_csvs/combined_CTU-13.csv")
    splits = split_data(CTU_13, label_column="BinaryLabel")

    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    X_explain, y_explain = splits["explain"]

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Explain set: {len(X_explain)} samples")

    np.save("CTU-13_npys/All/X_train.npy", X_train)
    np.save("CTU-13_npys/All/X_test.npy", X_test)
    np.save("CTU-13_npys/All/Y_train.npy", y_train)
    np.save("CTU-13_npys/All/Y_test.npy", y_test)
    np.save("CTU-13_npys/All/X_explain.npy", X_explain)
    np.save("CTU-13_npys/All/Y_explain.npy", y_explain)