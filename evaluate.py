import numpy as np
import typing as tp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import torch

from utils import Genre


def evaluate_model(y_true: torch.Tensor, y_pred: torch.Tensor):
    # calculate accuracy
    y_true, y_pred = y_true.numpy(), y_pred.numpy()
    num_correct = np.sum(y_true == y_pred)
    accuracy = num_correct / len(y_true)
    # calculate recall
    num_classes = len(Genre)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    for i in range(num_classes):
        confusion_matrix[i] /= np.sum(confusion_matrix[i])
    recall = np.mean(
        [
            confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            for i in range(num_classes)
        ]
    )
    # plot confusion matrix
    df_cm = pd.DataFrame(
        confusion_matrix, index=[i for i in Genre], columns=[i for i in Genre]
    )
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f')
    ax.xaxis.tick_top()
    plt.title(f"Accuracy: {accuracy:.2f}, Recall: {recall:.2f}")
    plt.show()


def get_model_accuracy(y_true: tp.List[int], y_pred: tp.List[int]) -> float:
    num_correct = np.sum(y_true == y_pred)
    accuracy = num_correct / len(y_true)
    return accuracy


if __name__ == '__main__':
    y = np.random.choice([0, 1, 2], size=32)
    y_hat = np.random.choice([0, 1, 2], size=32)
    evaluate_model(y, y_hat)
