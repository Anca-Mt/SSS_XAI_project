import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

valid_datasets = [
    "CTU-13",
    "NSL-KDD",
    "UNSW-NB15",
]

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Compute Accuracy, Precision, Recall, F1-score')

    argparser.add_argument('-d', '--dataset', default='NSL-KDD', type=str, help='The datset to split as a string')
    argparser.add_argument('--classifier', default="ebmclassifier", type=str, help='The classifier as a string.')

    args = argparser.parse_args()

    if args.dataset is None:
        raise ValueError(f"Dataset argument not provided. Use the -d or --dataset flag to provide one of {valid_datasets}.")

    print(f"The evaluated dataset is {args.dataset} using classifier {args.classifier}.")
    pred_vs_true_path = f"results/{args.dataset}/pred_vs_true_{args.classifier}.csv"

    if not os.path.exists(pred_vs_true_path):
        raise FileNotFoundError(f"File not found: {pred_vs_true_path}")

    pred_df = pd.read_csv(
        pred_vs_true_path,
        header=None,
        names=["index", "y_pred_raw", "y_true_raw", "correct"]
    )

    # Extract y_pred and y_true
    pred_df["y_pred"] = pred_df["y_pred_raw"].str.split(": ").str[-1].astype(int)
    pred_df["y_true"] = pred_df["y_true_raw"].str.split(": ").str[-1].astype(int)

    y_pred = pred_df["y_pred"]
    y_true = pred_df["y_true"]

    compute_metrics(y_true=y_true, y_pred=y_pred)
